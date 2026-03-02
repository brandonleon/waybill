from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Optional

from .models import ItemInfo, MarketOrder, StationRef, SystemInfo

ESI_BASE = "https://esi.evetech.net/latest"
DEFAULT_CACHE_TTL = 300


@dataclass(frozen=True)
class HttpResponse:
    data: Any
    headers: dict[str, str]
    status: int
    cached: bool
    fetched_at: int
    expires_at: int


class EsiClient:
    """Minimal ESI client with SQLite-backed response caching."""

    def __init__(self, cache_path: str, user_agent: str = "waybill/0.1") -> None:
        self.cache_path = cache_path
        self.user_agent = user_agent
        self.last_region_fetch: dict[int, dict[str, int]] = {}
        self._ensure_cache_dir()
        self._init_db()

    def _ensure_cache_dir(self) -> None:
        directory = os.path.dirname(self.cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS http_cache (
                    cache_key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    method TEXT NOT NULL,
                    body_hash TEXT,
                    response_json TEXT NOT NULL,
                    headers_json TEXT NOT NULL,
                    etag TEXT,
                    status INTEGER NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS item_types (
                    type_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    volume REAL NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_item_types_updated_at ON item_types(updated_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_orders (
                    order_id INTEGER PRIMARY KEY,
                    region_id INTEGER NOT NULL,
                    type_id INTEGER NOT NULL,
                    location_id INTEGER NOT NULL,
                    is_buy_order INTEGER NOT NULL,
                    price REAL NOT NULL,
                    volume_remain INTEGER NOT NULL,
                    min_volume INTEGER NOT NULL,
                    issued TEXT,
                    range TEXT,
                    duration INTEGER,
                    fetched_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_orders_region ON market_orders(region_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_orders_location ON market_orders(location_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_orders_type ON market_orders(type_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS region_names (
                    region_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS region_fetch_times (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_id INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    pages INTEGER NOT NULL,
                    cached INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_region_fetch_times_region
                ON region_fetch_times(region_id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_info (
                    system_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    security REAL NOT NULL,
                    region_id INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_neighbors (
                    system_id INTEGER NOT NULL,
                    neighbor_id INTEGER NOT NULL,
                    PRIMARY KEY (system_id, neighbor_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS station_names (
                    station_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def get_region_cache_meta(
        self, region_id: int, pages_limit: Optional[int] = None
    ) -> dict[str, Optional[int]]:
        pattern = f"{ESI_BASE}/markets/{region_id}/orders/%"
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            if pages_limit is None:
                row = conn.execute(
                    """
                    SELECT
                        MIN(expires_at) AS min_expires_at,
                        MAX(fetched_at) AS max_fetched_at,
                        COUNT(*) AS rows_count
                    FROM http_cache
                    WHERE method = 'GET' AND url LIKE ?
                    """,
                    (pattern,),
                ).fetchone()
                if not row:
                    return {"min_expires_at": None, "max_fetched_at": None, "rows_count": 0}
                return {
                    "min_expires_at": row["min_expires_at"],
                    "max_fetched_at": row["max_fetched_at"],
                    "rows_count": row["rows_count"],
                }
            rows = conn.execute(
                """
                SELECT url, expires_at, fetched_at
                FROM http_cache
                WHERE method = 'GET' AND url LIKE ?
                """,
                (pattern,),
            ).fetchall()

        if not rows:
            return {"min_expires_at": None, "max_fetched_at": None, "rows_count": 0}

        def extract_page(url: str) -> Optional[int]:
            try:
                query = urllib.parse.urlparse(url).query
                page = (urllib.parse.parse_qs(query).get("page") or [None])[0]
                return int(page) if page is not None else None
            except (TypeError, ValueError):
                return None

        min_expires_at: Optional[int] = None
        max_fetched_at: Optional[int] = None
        rows_count = 0
        for row in rows:
            page_num = extract_page(row["url"])
            if page_num is None or page_num > int(pages_limit):
                continue
            rows_count += 1
            expires_at = int(row["expires_at"])
            fetched_at = int(row["fetched_at"])
            if min_expires_at is None or expires_at < min_expires_at:
                min_expires_at = expires_at
            if max_fetched_at is None or fetched_at > max_fetched_at:
                max_fetched_at = fetched_at

        if rows_count == 0:
            return {"min_expires_at": None, "max_fetched_at": None, "rows_count": 0}

        return {
            "min_expires_at": min_expires_at,
            "max_fetched_at": max_fetched_at,
            "rows_count": rows_count,
        }

    def get_region_last_fetch(self, region_id: int) -> dict[str, Optional[int]]:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT fetched_at, pages, cached
                FROM region_fetch_times
                WHERE region_id = ?
                ORDER BY fetched_at DESC
                LIMIT 1
                """,
                (region_id,),
            ).fetchone()
        if not row:
            return {"fetched_at": None, "pages": None, "cached": None}
        return {
            "fetched_at": int(row["fetched_at"]),
            "pages": int(row["pages"]),
            "cached": int(row["cached"]),
        }

    def get_region_fetch_average(
        self,
        region_id: int,
        cached: Optional[bool] = None,
        limit: int = 20,
    ) -> tuple[Optional[float], int]:
        where_cached = ""
        params: list[Any] = [region_id]
        if cached is not None:
            where_cached = "AND cached = ?"
            params.append(1 if cached else 0)
        params.append(limit)
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT duration_ms
                FROM region_fetch_times
                WHERE region_id = ? {where_cached}
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        if not rows:
            return None, 0
        total = sum(int(row["duration_ms"]) for row in rows)
        avg_ms = total / len(rows)
        return avg_ms / 1000.0, len(rows)

    def _record_region_fetch_time(
        self,
        region_id: int,
        duration_ms: int,
        fetched_at: int,
        pages: int,
        cached: bool,
    ) -> None:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO region_fetch_times (
                    region_id, duration_ms, fetched_at, pages, cached
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (region_id, duration_ms, fetched_at, pages, 1 if cached else 0),
            )
            conn.commit()

    def _cache_key(self, method: str, url: str, body: Optional[bytes]) -> str:
        body_hash = hashlib.sha256(body or b"").hexdigest()
        key_raw = f"{method.upper()} {url} {body_hash}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def _load_cache(self, cache_key: str) -> Optional[dict[str, Any]]:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM http_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def _save_cache(
        self,
        cache_key: str,
        url: str,
        method: str,
        body_hash: str,
        response_json: str,
        headers_json: str,
        etag: Optional[str],
        status: int,
        fetched_at: int,
        expires_at: int,
    ) -> None:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO http_cache (
                    cache_key, url, method, body_hash, response_json, headers_json,
                    etag, status, fetched_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    headers_json = excluded.headers_json,
                    etag = excluded.etag,
                    status = excluded.status,
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    url,
                    method,
                    body_hash,
                    response_json,
                    headers_json,
                    etag,
                    status,
                    fetched_at,
                    expires_at,
                ),
            )
            conn.commit()

    def _parse_expiry(self, headers: dict[str, str], fetched_at: int) -> int:
        cache_control = headers.get("Cache-Control", "")
        for part in cache_control.split(","):
            part = part.strip()
            if part.startswith("max-age="):
                try:
                    seconds = int(part.split("=", 1)[1])
                    return fetched_at + seconds
                except ValueError:
                    continue

        expires_header = headers.get("Expires")
        if expires_header:
            try:
                expires_dt = parsedate_to_datetime(expires_header)
                if expires_dt.tzinfo is None:
                    expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                return int(expires_dt.timestamp())
            except (TypeError, ValueError):
                pass

        return fetched_at + DEFAULT_CACHE_TTL

    def _request_json(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        ttl_override: Optional[int] = None,
        force_refresh: bool = False,
    ) -> HttpResponse:
        method = method.upper()
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = f"{ESI_BASE}{path}"
        if query:
            url = f"{url}?{query}"

        body_bytes: Optional[bytes] = None
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")

        cache_key = self._cache_key(method, url, body_bytes)
        cached = None if force_refresh else self._load_cache(cache_key)

        now = int(time.time())
        if cached and cached["expires_at"] > now:
            cached_headers = json.loads(cached.get("headers_json") or "{}")
            return HttpResponse(
                data=json.loads(cached["response_json"]),
                headers=cached_headers,
                status=int(cached["status"]),
                cached=True,
                fetched_at=int(cached["fetched_at"]),
                expires_at=int(cached["expires_at"]),
            )

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if cached and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]

        request = urllib.request.Request(url, data=body_bytes, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    import gzip

                    raw = gzip.decompress(raw)

                text = raw.decode("utf-8")
                data = json.loads(text) if text else None
                fetched_at = int(time.time())
                if ttl_override is not None:
                    expires_at = fetched_at + ttl_override
                else:
                    expires_at = self._parse_expiry(dict(response.headers), fetched_at)
                etag = response.headers.get("ETag")
                body_hash = hashlib.sha256(body_bytes or b"").hexdigest()
                self._save_cache(
                    cache_key,
                    url,
                    method,
                    body_hash,
                    json.dumps(data),
                    json.dumps(dict(response.headers)),
                    etag,
                    response.status,
                    fetched_at,
                    expires_at,
                )
                return HttpResponse(
                    data=data,
                    headers=dict(response.headers),
                    status=response.status,
                    cached=False,
                    fetched_at=fetched_at,
                    expires_at=expires_at,
                )
        except urllib.error.HTTPError as exc:
            if exc.code == 304 and cached and not force_refresh:
                fetched_at = int(time.time())
                if ttl_override is not None:
                    expires_at = fetched_at + ttl_override
                else:
                    expires_at = self._parse_expiry(dict(exc.headers), fetched_at)
                body_hash = hashlib.sha256(body_bytes or b"").hexdigest()
                self._save_cache(
                    cache_key,
                    url,
                    method,
                    body_hash,
                    cached["response_json"],
                    cached.get("headers_json") or "{}",
                    cached.get("etag"),
                    cached.get("status", 200),
                    fetched_at,
                    expires_at,
                )
                return HttpResponse(
                    data=json.loads(cached["response_json"]),
                    headers=dict(exc.headers),
                    status=cached.get("status", 200),
                    cached=True,
                    fetched_at=fetched_at,
                    expires_at=expires_at,
                )
            raise

    def resolve_station(
        self, name: str, allow_structures: bool = False
    ) -> StationRef:
        """Resolve a station name to an NPC station id and its region."""
        ids = self._request_json("POST", "/universe/ids/", json_body=[name]).data
        stations = ids.get("stations") or []
        structures = ids.get("structures") or []
        systems = ids.get("systems") or []

        if stations:
            station_id = stations[0]["id"]
            return self._station_ref(station_id)

        if structures:
            if not allow_structures:
                raise ValueError(
                    f"'{name}' resolved to a player structure. Use --allow-structures to allow it."
                )
            raise ValueError(
                "Player structures require authenticated ESI endpoints and are not supported."
            )

        if systems:
            system_id = systems[0]["id"]
            station_ids = self._request_json(
                "GET", f"/universe/systems/{system_id}/"
            ).data.get("stations", [])
            if len(station_ids) == 1:
                return self._station_ref(station_ids[0])
            if station_ids:
                names = self.get_names(station_ids[:10])
                formatted = ", ".join(names)
                raise ValueError(
                    f"System '{name}' has multiple stations. Provide a full station name"
                    f" or use --from-system/--to-system. Examples: {formatted}"
                )

        try:
            search = self._request_json(
                "GET",
                "/search/",
                params={"categories": "station", "search": name, "strict": "false"},
            ).data
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            search = None
        station_ids = search.get("station", []) if search else []
        if not station_ids:
            raise ValueError(f"No station found matching '{name}'.")
        if len(station_ids) > 1:
            names = self.get_names(station_ids[:10])
            formatted = ", ".join(names)
            raise ValueError(
                f"Multiple stations match '{name}'. Provide a full station name."
                f" Examples: {formatted}"
            )
        return self._station_ref(station_ids[0])

    def resolve_system_id(self, name: str) -> int:
        return self._resolve_system_id(name)

    def resolve_region_id(self, name: str) -> int:
        """Resolve a region name to its region_id via ESI."""
        ids = self._request_json("POST", "/universe/ids/", json_body=[name]).data
        regions = ids.get("regions") or []
        if not regions:
            raise ValueError(
                f"No region found matching '{name}'. Provide the exact region name"
                " (e.g. 'The Forge', 'Metropolis', 'Domain')."
            )
        return int(regions[0]["id"])

    def _station_ref(self, station_id: int) -> StationRef:
        station = self._request_json("GET", f"/universe/stations/{station_id}/").data
        system_id = station["system_id"]
        region_id = self._system_region_id(system_id)
        return StationRef(
            station_id=station_id,
            name=station.get("name", str(station_id)),
            system_id=system_id,
            region_id=region_id,
        )

    def resolve_system_stations(self, name: str) -> list[StationRef]:
        """Resolve a system name and return all NPC stations within it."""
        system_id = self._resolve_system_id(name)
        system = self._request_json("GET", f"/universe/systems/{system_id}/").data
        station_ids = system.get("stations", [])
        if not station_ids:
            raise ValueError(f"System '{name}' has no NPC stations.")

        region_id = self._system_region_id(system_id)
        stations: list[StationRef] = []
        for station_id in station_ids:
            station = self._request_json(
                "GET", f"/universe/stations/{station_id}/"
            ).data
            stations.append(
                StationRef(
                    station_id=station_id,
                    name=station.get("name", str(station_id)),
                    system_id=system_id,
                    region_id=region_id,
                )
            )
        return stations

    def _resolve_system_id(self, name: str) -> int:
        ids = self._request_json("POST", "/universe/ids/", json_body=[name]).data
        systems = ids.get("systems") or []
        if systems:
            return systems[0]["id"]

        # ESI search uses the "solar_system" category (not "system").
        try:
            search = self._request_json(
                "GET",
                "/search/",
                params={"categories": "solar_system", "search": name, "strict": "false"},
            ).data
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            search = None
        system_ids = search.get("system", []) if search else []
        if not system_ids:
            raise ValueError(f"No system found matching '{name}'.")
        if len(system_ids) > 1:
            names = self.get_names(system_ids[:10])
            formatted = ", ".join(names)
            raise ValueError(
                f"Multiple systems match '{name}'. Provide a full system name."
                f" Examples: {formatted}"
            )
        return system_ids[0]

    def _system_region_id(self, system_id: int) -> int:
        system = self._request_json("GET", f"/universe/systems/{system_id}/").data
        constellation_id = system["constellation_id"]
        constellation = self._request_json(
            "GET", f"/universe/constellations/{constellation_id}/"
        ).data
        return int(constellation["region_id"])

    def get_names(self, ids: Iterable[int]) -> list[str]:
        payload = list(ids)
        if not payload:
            return []
        data = self._request_json("POST", "/universe/names/", json_body=payload).data
        if not data:
            return []
        return [entry["name"] for entry in data if "name" in entry]

    def get_station_names(self, station_ids: Iterable[int]) -> dict[int, str]:
        ids = list(dict.fromkeys(station_ids))
        if not ids:
            return {}

        names: dict[int, str] = {}
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            for offset in range(0, len(ids), 900):
                chunk = ids[offset : offset + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT station_id, name FROM station_names WHERE station_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    names[int(row["station_id"])] = row["name"]

        missing = [station_id for station_id in ids if station_id not in names]
        if missing:
            data = self._request_json("POST", "/universe/names/", json_body=missing).data or []
            fetched_at = int(time.time())
            with sqlite3.connect(self.cache_path, timeout=30) as conn:
                for entry in data:
                    if entry.get("category") == "station":
                        station_id = int(entry["id"])
                        station_name = entry["name"]
                        names[station_id] = station_name
                        conn.execute(
                            """
                            INSERT INTO station_names (station_id, name, updated_at)
                            VALUES (?, ?, ?)
                            ON CONFLICT(station_id) DO UPDATE SET
                                name = excluded.name,
                                updated_at = excluded.updated_at
                            """,
                            (station_id, station_name, fetched_at),
                        )
                conn.commit()

        return names

    def get_system_info(self, system_id: int) -> SystemInfo:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM system_info WHERE system_id = ?",
                (system_id,),
            ).fetchone()
            if row:
                return SystemInfo(
                    system_id=system_id,
                    name=row["name"],
                    security=float(row["security"]),
                    region_id=int(row["region_id"]),
                )

        system = self._request_json("GET", f"/universe/systems/{system_id}/").data
        name = system.get("name", str(system_id))
        security = float(system.get("security_status", 0.0))
        constellation_id = system["constellation_id"]
        constellation = self._request_json(
            "GET", f"/universe/constellations/{constellation_id}/"
        ).data
        region_id = int(constellation["region_id"])
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO system_info (system_id, name, security, region_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(system_id) DO UPDATE SET
                    name = excluded.name,
                    security = excluded.security,
                    region_id = excluded.region_id,
                    updated_at = excluded.updated_at
                """,
                (system_id, name, security, region_id, int(time.time())),
            )
            conn.commit()

        return SystemInfo(
            system_id=system_id, name=name, security=security, region_id=region_id
        )

    def get_system_station_ids(self, system_id: int) -> list[int]:
        system = self._request_json("GET", f"/universe/systems/{system_id}/").data
        return [int(station_id) for station_id in system.get("stations", [])]

    def get_system_neighbors(self, system_id: int) -> list[int]:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            rows = conn.execute(
                "SELECT neighbor_id FROM system_neighbors WHERE system_id = ?",
                (system_id,),
            ).fetchall()
            if rows:
                return [int(row[0]) for row in rows]

        system = self._request_json("GET", f"/universe/systems/{system_id}/").data
        neighbors: set[int] = set()
        for gate_id in system.get("stargates", []) or []:
            gate = self._request_json("GET", f"/universe/stargates/{gate_id}/").data
            dest = gate.get("destination", {})
            neighbor_id = dest.get("system_id")
            if neighbor_id is not None:
                neighbors.add(int(neighbor_id))

        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO system_neighbors (system_id, neighbor_id)
                VALUES (?, ?)
                """,
                [(system_id, neighbor_id) for neighbor_id in neighbors],
            )
            conn.commit()

        return sorted(neighbors)

    def get_type_info(self, type_id: int) -> ItemInfo:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM item_types WHERE type_id = ?", (type_id,)
            ).fetchone()
            if row:
                return ItemInfo(
                    type_id=type_id, name=row["name"], volume=row["volume"]
                )

        data = self._request_json("GET", f"/universe/types/{type_id}/").data
        info = ItemInfo(type_id=type_id, name=data["name"], volume=data["volume"])
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO item_types (type_id, name, volume, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(type_id) DO UPDATE SET
                    name = excluded.name,
                    volume = excluded.volume,
                    updated_at = excluded.updated_at
                """,
                (type_id, info.name, info.volume, int(time.time())),
            )
            conn.commit()
        return info

    def cache_type_info_bulk(
        self,
        type_ids: Iterable[int],
        refresh: bool = False,
        workers: int = 1,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[int, int, int]:
        ids = [int(type_id) for type_id in dict.fromkeys(type_ids)]
        total = len(ids)
        if total == 0:
            return 0, 0, 0

        existing: set[int] = set()
        if not refresh:
            with sqlite3.connect(self.cache_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row
                for offset in range(0, len(ids), 900):
                    chunk = ids[offset : offset + 900]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT type_id FROM item_types WHERE type_id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for row in rows:
                        existing.add(int(row["type_id"]))

        to_fetch = ids if refresh else [type_id for type_id in ids if type_id not in existing]
        skipped = total - len(to_fetch)
        if not to_fetch:
            return 0, skipped, 0

        results: list[tuple[int, str, float]] = []
        errors = 0
        completed = 0

        def handle_result(type_id: int, data: Any) -> None:
            nonlocal completed
            results.append((type_id, data["name"], float(data["volume"])))
            completed += 1
            if progress:
                progress(completed, len(to_fetch))

        def fetch_one(type_id: int) -> tuple[int, Any]:
            data = self._request_json("GET", f"/universe/types/{type_id}/").data
            return type_id, data

        if workers and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(fetch_one, type_id): type_id for type_id in to_fetch}
                for future in as_completed(futures):
                    try:
                        type_id, data = future.result()
                        handle_result(type_id, data)
                    except Exception:
                        errors += 1
        else:
            for type_id in to_fetch:
                try:
                    type_id, data = fetch_one(type_id)
                    handle_result(type_id, data)
                except Exception:
                    errors += 1

        if results:
            now = int(time.time())
            with sqlite3.connect(self.cache_path, timeout=30) as conn:
                conn.executemany(
                    """
                    INSERT INTO item_types (type_id, name, volume, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(type_id) DO UPDATE SET
                        name = excluded.name,
                        volume = excluded.volume,
                        updated_at = excluded.updated_at
                    """,
                    [(type_id, name, volume, now) for type_id, name, volume in results],
                )
                conn.commit()

        return len(results), skipped, errors

    def refresh_oldest_type_info(
        self,
        limit: int = 10,
        max_age_seconds: int = 0,
        workers: int = 2,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[int, int, int]:
        if limit <= 0:
            return 0, 0, 0

        where_clause = ""
        params: list[Any] = []
        if max_age_seconds and max_age_seconds > 0:
            cutoff = int(time.time()) - int(max_age_seconds)
            where_clause = "WHERE updated_at <= ?"
            params.append(cutoff)

        params.append(limit)
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT type_id FROM item_types {where_clause} "
                "ORDER BY updated_at ASC LIMIT ?",
                params,
            ).fetchall()

        type_ids = [int(row["type_id"]) for row in rows]
        if not type_ids:
            return 0, 0, 0

        return self.cache_type_info_bulk(
            type_ids,
            refresh=True,
            workers=workers,
            progress=progress,
        )

    def get_region_name(self, region_id: int) -> str:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT name FROM region_names WHERE region_id = ?",
                (region_id,),
            ).fetchone()
            if row:
                return row["name"]

        data = self._request_json("GET", f"/universe/regions/{region_id}/").data
        if not data:
            return str(region_id)
        name = data.get("name", str(region_id))
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO region_names (region_id, name, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(region_id) DO UPDATE SET
                    name = excluded.name,
                    updated_at = excluded.updated_at
                """,
                (region_id, name, int(time.time())),
            )
            conn.commit()
        return name

    def resolve_type_id(self, name: str) -> int:
        ids = self._request_json("POST", "/universe/ids/", json_body=[name]).data
        types = ids.get("inventory_types") or ids.get("types") or []
        if types:
            return int(types[0]["id"])

        try:
            search = self._request_json(
                "GET",
                "/search/",
                params={"categories": "inventory_type", "search": name, "strict": "false"},
            ).data
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            search = None
        type_ids = search.get("inventory_type", []) if search else []
        if not type_ids:
            raise ValueError(f"No item found matching '{name}'.")
        if len(type_ids) > 1:
            names = self.get_names(type_ids[:10])
            formatted = ", ".join(names)
            raise ValueError(
                f"Multiple items match '{name}'. Provide a full item name."
                f" Examples: {formatted}"
            )
        return int(type_ids[0])

    def fetch_region_orders(
        self,
        region_id: int,
        cache_ttl: int = 1800,
        progress: Optional[Callable[[int, int], None]] = None,
        page_workers: int = 1,
        force_refresh: bool = False,
    ) -> list[MarketOrder]:
        start_time = time.monotonic()
        orders: list[MarketOrder] = []
        page = 1
        pages = 1
        fetched_at = int(time.time())
        latest_fetched_at = 0
        earliest_expires_at: Optional[int] = None
        all_cached = True
        cached_pages = 0
        fetched_pages = 0

        def process_response(response: HttpResponse, page_num: int) -> None:
            nonlocal latest_fetched_at, earliest_expires_at, all_cached, cached_pages, fetched_pages
            data = response.data or []
            if progress:
                progress(page_num, pages)
            latest_fetched_at = max(latest_fetched_at, response.fetched_at)
            if earliest_expires_at is None or response.expires_at < earliest_expires_at:
                earliest_expires_at = response.expires_at
            all_cached = all_cached and response.cached
            if response.cached:
                cached_pages += 1
            else:
                fetched_pages += 1
            for entry in data:
                is_buy = bool(entry.get("is_buy_order"))
                orders.append(
                    MarketOrder(
                        order_id=int(entry["order_id"]),
                        type_id=entry["type_id"],
                        location_id=entry["location_id"],
                        is_buy_order=is_buy,
                        price=float(entry["price"]),
                        volume_remain=int(entry["volume_remain"]),
                        min_volume=int(entry.get("min_volume", 1)),
                        issued=entry.get("issued"),
                        order_range=entry.get("range"),
                        duration=entry.get("duration"),
                    )
                )

        ttl_override = cache_ttl if cache_ttl > 0 else None
        first_response = self._request_json(
            "GET",
            f"/markets/{region_id}/orders/",
            params={"order_type": "all", "page": page},
            ttl_override=ttl_override,
            force_refresh=force_refresh,
        )
        pages = (
            int(first_response.headers.get("X-Pages", pages))
            if first_response.headers
            else pages
        )
        process_response(first_response, page)

        if pages > 1:
            if page_workers and page_workers > 1:
                with ThreadPoolExecutor(max_workers=page_workers) as executor:
                    futures = {
                        executor.submit(
                            self._request_json,
                            "GET",
                            f"/markets/{region_id}/orders/",
                            {"order_type": "all", "page": page_num},
                            None,
                            ttl_override,
                            force_refresh,
                        ): page_num
                        for page_num in range(2, pages + 1)
                    }
                    for future in as_completed(futures):
                        page_num = futures[future]
                        response = future.result()
                        process_response(response, page_num)
            else:
                for page_num in range(2, pages + 1):
                    response = self._request_json(
                        "GET",
                        f"/markets/{region_id}/orders/",
                        params={"order_type": "all", "page": page_num},
                        ttl_override=ttl_override,
                        force_refresh=force_refresh,
                    )
                    process_response(response, page_num)
        if earliest_expires_at is None:
            earliest_expires_at = fetched_at
        self.last_region_fetch[region_id] = {
            "fetched_at": latest_fetched_at or fetched_at,
            "expires_at": earliest_expires_at,
            "pages": pages,
            "cached": all_cached,
            "cached_pages": cached_pages,
            "fetched_pages": fetched_pages,
        }
        self._store_market_orders(region_id, orders, fetched_at)
        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._record_region_fetch_time(
            region_id,
            duration_ms,
            int(time.time()),
            pages,
            all_cached,
        )
        return orders

    def fetch_item_orders(
        self,
        region_id: int,
        type_id: int,
        cache_ttl: int = 1800,
    ) -> list[MarketOrder]:
        """Fetch market orders for a single item type in a region.

        Uses ESI's type_id filter so only orders for that item are downloaded,
        instead of pulling the entire region order book.
        """
        orders: list[MarketOrder] = []
        page = 1
        pages = 1
        ttl_override = cache_ttl if cache_ttl > 0 else None

        def fetch_page(page_num: int) -> HttpResponse:
            return self._request_json(
                "GET",
                f"/markets/{region_id}/orders/",
                params={"order_type": "all", "type_id": type_id, "page": page_num},
                ttl_override=ttl_override,
            )

        def parse_response(response: HttpResponse) -> None:
            for entry in response.data or []:
                is_buy = bool(entry.get("is_buy_order"))
                orders.append(
                    MarketOrder(
                        order_id=int(entry["order_id"]),
                        type_id=entry["type_id"],
                        location_id=entry["location_id"],
                        is_buy_order=is_buy,
                        price=float(entry["price"]),
                        volume_remain=int(entry["volume_remain"]),
                        min_volume=int(entry.get("min_volume", 1)),
                        issued=entry.get("issued"),
                        order_range=entry.get("range"),
                        duration=entry.get("duration"),
                    )
                )

        first = fetch_page(page)
        pages = int(first.headers.get("X-Pages", 1)) if first.headers else 1
        parse_response(first)

        for page_num in range(2, pages + 1):
            parse_response(fetch_page(page_num))

        return orders

    def _store_market_orders(
        self, region_id: int, orders: list[MarketOrder], fetched_at: int
    ) -> None:
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute("DELETE FROM market_orders WHERE region_id = ?", (region_id,))
            if orders:
                conn.executemany(
                    """
                    INSERT INTO market_orders (
                        order_id, region_id, type_id, location_id, is_buy_order,
                        price, volume_remain, min_volume, issued, range, duration, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(order_id) DO UPDATE SET
                        region_id = excluded.region_id,
                        type_id = excluded.type_id,
                        location_id = excluded.location_id,
                        is_buy_order = excluded.is_buy_order,
                        price = excluded.price,
                        volume_remain = excluded.volume_remain,
                        min_volume = excluded.min_volume,
                        issued = excluded.issued,
                        range = excluded.range,
                        duration = excluded.duration,
                        fetched_at = excluded.fetched_at
                    """,
                    [
                        (
                            order.order_id,
                            region_id,
                            order.type_id,
                            order.location_id,
                            1 if order.is_buy_order else 0,
                            order.price,
                            order.volume_remain,
                            order.min_volume,
                            order.issued,
                            order.order_range,
                            order.duration,
                            fetched_at,
                        )
                        for order in orders
                    ],
                )
            conn.commit()
