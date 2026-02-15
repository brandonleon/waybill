from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .models import ItemInfo, MarketOrder, Opportunity, SystemInfo


class OrderBook:
    """Best-price snapshots for orders at a station."""

    def __init__(self) -> None:
        self._best_price: dict[int, float] = {}
        self._volume_at_best: dict[int, int] = {}

    def update_best(self, type_id: int, price: float, volume: int, prefer_min: bool) -> None:
        if type_id not in self._best_price:
            self._best_price[type_id] = price
            self._volume_at_best[type_id] = volume
            return

        current_price = self._best_price[type_id]
        if prefer_min:
            if price < current_price:
                self._best_price[type_id] = price
                self._volume_at_best[type_id] = volume
            elif price == current_price:
                self._volume_at_best[type_id] += volume
        else:
            if price > current_price:
                self._best_price[type_id] = price
                self._volume_at_best[type_id] = volume
            elif price == current_price:
                self._volume_at_best[type_id] += volume

    def best_price(self, type_id: int) -> float | None:
        return self._best_price.get(type_id)

    def volume_at_best(self, type_id: int) -> int:
        return self._volume_at_best.get(type_id, 0)

    def type_ids(self) -> Iterable[int]:
        return self._best_price.keys()


@dataclass(frozen=True)
class BestStationPrice:
    price: float
    volume: int
    station_id: int


def best_station_prices(
    orders: Iterable[MarketOrder],
    station_ids: Iterable[int],
    *,
    is_buy_order: bool,
    prefer_min: bool,
    type_filter: Optional[set[int]] = None,
) -> dict[int, BestStationPrice]:
    """Return best price per type across stations, preserving the station id."""
    station_set = set(station_ids)
    station_books: dict[int, OrderBook] = {}

    for order in orders:
        if order.location_id not in station_set:
            continue
        if order.is_buy_order != is_buy_order:
            continue
        if type_filter is not None and order.type_id not in type_filter:
            continue
        book = station_books.setdefault(order.location_id, OrderBook())
        book.update_best(order.type_id, order.price, order.volume_remain, prefer_min=prefer_min)

    best: dict[int, BestStationPrice] = {}
    for station_id, book in station_books.items():
        for type_id in book.type_ids():
            price = book.best_price(type_id)
            if price is None:
                continue
            volume = book.volume_at_best(type_id)
            current = best.get(type_id)
            if current is None:
                best[type_id] = BestStationPrice(price=price, volume=volume, station_id=station_id)
                continue
            if prefer_min:
                if price < current.price or (price == current.price and volume > current.volume):
                    best[type_id] = BestStationPrice(price=price, volume=volume, station_id=station_id)
            else:
                if price > current.price or (price == current.price and volume > current.volume):
                    best[type_id] = BestStationPrice(price=price, volume=volume, station_id=station_id)

    return best


def build_order_books(
    orders: Iterable[MarketOrder],
    source_station_ids: Iterable[int],
    destination_station_ids: Iterable[int],
) -> tuple[OrderBook, OrderBook, OrderBook, OrderBook, dict[str, dict[int, dict[str, int]]]]:
    """Return best-price order snapshots for source/destination buys and sells."""
    source_sells = OrderBook()
    source_buys = OrderBook()
    destination_sells = OrderBook()
    destination_buys = OrderBook()
    stats: dict[str, dict[int, dict[str, int]]] = {
        "source": {},
        "destination": {},
    }

    source_set = set(source_station_ids)
    dest_set = set(destination_station_ids)

    def bump(bucket: str, station_id: int, is_buy: bool) -> None:
        entry = stats[bucket].setdefault(
            station_id, {"total": 0, "buy": 0, "sell": 0}
        )
        entry["total"] += 1
        if is_buy:
            entry["buy"] += 1
        else:
            entry["sell"] += 1

    for order in orders:
        if order.location_id in source_set:
            bump("source", order.location_id, order.is_buy_order)
            if order.is_buy_order:
                source_buys.update_best(
                    order.type_id, order.price, order.volume_remain, prefer_min=False
                )
            else:
                source_sells.update_best(
                    order.type_id, order.price, order.volume_remain, prefer_min=True
                )
        elif order.location_id in dest_set:
            bump("destination", order.location_id, order.is_buy_order)
            if order.is_buy_order:
                destination_buys.update_best(
                    order.type_id, order.price, order.volume_remain, prefer_min=False
                )
            else:
                destination_sells.update_best(
                    order.type_id, order.price, order.volume_remain, prefer_min=True
                )

    return source_sells, source_buys, destination_sells, destination_buys, stats


def compute_opportunities(
    source_sells: OrderBook,
    source_buys: OrderBook,
    destination_sells: OrderBook,
    destination_buys: OrderBook,
    type_info: Callable[[int], ItemInfo],
    cargo_m3: float,
    broker_fee: float,
    sales_tax: float,
    min_profit: float,
    min_cargo_fill_ratio: float,
    backhaul_mode: bool,
    instant_only: bool,
) -> list[Opportunity]:
    """Compute ranked opportunities based on best-price orders."""
    opportunities: list[Opportunity] = []

    def consider_pair(
        buy_book: OrderBook,
        sell_book: OrderBook,
        buy_order_type: str,
        sell_order_type: str,
    ) -> None:
        shared_type_ids = set(buy_book.type_ids()) & set(sell_book.type_ids())

        for type_id in shared_type_ids:
            buy_price = buy_book.best_price(type_id)
            sell_price = sell_book.best_price(type_id)
            if buy_price is None or sell_price is None:
                continue
            if sell_price <= buy_price:
                continue

            item = type_info(type_id)
            if item.volume <= 0:
                continue

            max_units = int(math.floor(cargo_m3 / item.volume))
            if max_units <= 0:
                continue

            sell_volume = sell_book.volume_at_best(type_id)
            buy_volume = buy_book.volume_at_best(type_id)
            quantity = min(max_units, sell_volume, buy_volume)
            if quantity <= 0:
                continue
            total_volume = quantity * item.volume
            fill_ratio = total_volume / cargo_m3 if cargo_m3 > 0 else 0.0
            if fill_ratio < min_cargo_fill_ratio:
                continue

            gross_profit_per_unit = sell_price - buy_price
            net_sell_price = sell_price * (1.0 - broker_fee - sales_tax)
            net_profit_per_unit = net_sell_price - buy_price
            if net_profit_per_unit <= 0:
                continue

            gross_profit_total = gross_profit_per_unit * quantity
            net_profit_total = net_profit_per_unit * quantity
            if net_profit_total < min_profit:
                continue

            isk_per_m3 = net_profit_per_unit / item.volume
            opportunities.append(
                Opportunity(
                    type_id=type_id,
                    item_name=item.name,
                    buy_order_type=buy_order_type,
                    sell_order_type=sell_order_type,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    gross_profit_per_unit=gross_profit_per_unit,
                    gross_profit_total=gross_profit_total,
                    net_profit_per_unit=net_profit_per_unit,
                    net_profit_total=net_profit_total,
                    isk_per_m3=isk_per_m3,
                    quantity=quantity,
                    total_volume=total_volume,
                    liquidity=min(sell_volume, buy_volume),
                )
            )

    consider_pair(source_sells, destination_buys, "sell", "buy")
    if not instant_only:
        consider_pair(source_buys, destination_sells, "buy", "sell")

    if backhaul_mode:
        opportunities.sort(
            key=lambda opp: (
                opp.total_volume,
                opp.net_profit_total,
                opp.isk_per_m3,
                opp.liquidity,
            ),
            reverse=True,
        )
    else:
        opportunities.sort(
            key=lambda opp: (opp.isk_per_m3, opp.liquidity), reverse=True
        )
    return opportunities


def systems_within_jumps(
    start_system_id: int,
    max_jumps: int,
    min_security: float,
    max_security: Optional[float],
    get_neighbors: Callable[[int], Iterable[int]],
    get_system_info: Callable[[int], SystemInfo],
) -> list[tuple[SystemInfo, int]]:
    """Return systems within N jumps with their info and distance."""
    visited: set[int] = {start_system_id}
    queue = deque([(start_system_id, 0)])
    results: list[tuple[SystemInfo, int]] = []

    while queue:
        system_id, jumps = queue.popleft()
        info = get_system_info(system_id)
        if info.security >= min_security and (
            max_security is None or info.security <= max_security
        ):
            results.append((info, jumps))

        if jumps >= max_jumps:
            continue

        for neighbor in get_neighbors(system_id):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, jumps + 1))

    return results


def shortest_path(
    start_system_id: int,
    end_system_id: int,
    get_neighbors: Callable[[int], Iterable[int]],
) -> list[int]:
    """Return the shortest path between two systems (inclusive)."""
    if start_system_id == end_system_id:
        return [start_system_id]

    visited: set[int] = {start_system_id}
    parent: dict[int, int] = {}
    queue = deque([start_system_id])

    while queue:
        current = queue.popleft()
        for neighbor in get_neighbors(current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = current
            if neighbor == end_system_id:
                path = [end_system_id]
                while path[-1] != start_system_id:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(neighbor)

    return []


def systems_within_route_jumps(
    route_system_ids: Iterable[int],
    max_off_route_jumps: int,
    get_neighbors: Callable[[int], Iterable[int]],
) -> dict[int, int]:
    """Return systems within N jumps of the route, with distance to route."""
    distances: dict[int, int] = {}
    queue = deque()

    for system_id in route_system_ids:
        if system_id in distances:
            continue
        distances[system_id] = 0
        queue.append(system_id)

    while queue:
        system_id = queue.popleft()
        distance = distances[system_id]
        if distance >= max_off_route_jumps:
            continue
        for neighbor in get_neighbors(system_id):
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append(neighbor)

    return distances


def best_prices_for_item(
    orders: Iterable[MarketOrder],
    station_ids: Iterable[int],
    type_id: int,
) -> dict[int, dict[str, float | int | None]]:
    """Return best buy/sell prices per station for a single type."""
    station_set = set(station_ids)
    results: dict[int, dict[str, float | int | None]] = {}

    for station_id in station_set:
        results[station_id] = {
            "buy_price": None,
            "buy_volume": 0,
            "sell_price": None,
            "sell_volume": 0,
        }

    for order in orders:
        if order.type_id != type_id:
            continue
        if order.location_id not in station_set:
            continue

        entry = results[order.location_id]
        if order.is_buy_order:
            current = entry["buy_price"]
            if current is None or order.price > current:
                entry["buy_price"] = order.price
                entry["buy_volume"] = order.volume_remain
            elif order.price == current:
                entry["buy_volume"] = int(entry["buy_volume"]) + order.volume_remain
        else:
            current = entry["sell_price"]
            if current is None or order.price < current:
                entry["sell_price"] = order.price
                entry["sell_volume"] = order.volume_remain
            elif order.price == current:
                entry["sell_volume"] = int(entry["sell_volume"]) + order.volume_remain

    return results
