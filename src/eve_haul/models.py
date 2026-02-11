from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketOrder:
    """Normalized market order representation from ESI."""

    order_id: int
    type_id: int
    location_id: int
    is_buy_order: bool
    price: float
    volume_remain: int
    min_volume: int
    issued: str | None = None
    order_range: str | None = None
    duration: int | None = None


@dataclass(frozen=True)
class ItemInfo:
    """Item metadata used for hauling decisions."""

    type_id: int
    name: str
    volume: float


@dataclass(frozen=True)
class StationRef:
    """Resolved station identity with location context."""

    station_id: int
    name: str
    system_id: int
    region_id: int


@dataclass(frozen=True)
class SystemInfo:
    """System metadata used for navigation and filtering."""

    system_id: int
    name: str
    security: float
    region_id: int


@dataclass(frozen=True)
class Opportunity:
    """Computed arbitrage opportunity between two stations."""

    type_id: int
    item_name: str
    buy_order_type: str
    sell_order_type: str
    buy_price: float
    sell_price: float
    gross_profit_per_unit: float
    gross_profit_total: float
    net_profit_per_unit: float
    net_profit_total: float
    isk_per_m3: float
    quantity: int
    total_volume: float
    liquidity: int
