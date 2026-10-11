"""Core data types for invoicer.

Everything here is a plain dataclass so that objects are cheap to build in
tests and easy to serialise. Money is kept as float throughout; rounding to
cents happens only at the edges (see pricing.round_money).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass(frozen=True)
class Item:
    """A catalogue item.

    sku         stock keeping unit, unique within a catalogue
    name        human readable name, printed on invoices
    unit_price  price of a single unit before discounts and tax
    taxable     whether sales tax applies to this item
    stock       units on hand; None means "not tracked"
    """

    sku: str
    name: str
    unit_price: float
    taxable: bool = True
    stock: Optional[int] = None

    def in_stock(self, qty: int) -> bool:
        if self.stock is None:
            return True
        return self.stock >= qty


@dataclass
class LineItem:
    """One line on an invoice: an item and how many units were ordered."""

    item: Item
    qty: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"quantity must be positive, got {self.qty}")


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str = ""
    tax_exempt: bool = False
    region: str = "default"


@dataclass
class Invoice:
    """An invoice is a customer plus an ordered list of line items.

    Totals are never stored on the invoice; they are computed by the pricing
    module on demand so that a catalogue price change is always reflected.
    """

    number: str
    customer: Customer
    issued: date
    lines: List[LineItem] = field(default_factory=list)
    coupon: Optional[str] = None

    def add(self, item: Item, qty: int, note: str = "") -> LineItem:
        for line in self.lines:
            if line.item.sku == item.sku and line.note == note:
                line.qty += qty
                return line
        line = LineItem(item=item, qty=qty, note=note)
        self.lines.append(line)
        return line

    def remove(self, sku: str) -> None:
        self.lines = [l for l in self.lines if l.item.sku != sku]

    def units(self) -> int:
        return sum(l.qty for l in self.lines)

    def is_empty(self) -> bool:
        return not self.lines
