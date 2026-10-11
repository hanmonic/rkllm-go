"""Loading and saving the item catalogue.

The catalogue is a CSV file with a header row:

    sku,name,unit_price,stock,taxable
    W-100,Widget,10.00,250,yes
    G-200,Gadget,4.50,,no

stock may be empty (not tracked). taxable accepts yes/no/true/false/1/0 and
defaults to yes when the column is missing or empty.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, Iterable, List, Optional

from .models import Item

COL_SKU = 0
COL_NAME = 1
COL_PRICE = 2
COL_STOCK = 3
COL_TAXABLE = 4

HEADER = ["sku", "name", "unit_price", "stock", "taxable"]

TRUE_WORDS = {"yes", "y", "true", "1"}
FALSE_WORDS = {"no", "n", "false", "0"}


class CatalogError(ValueError):
    """Raised for malformed catalogue rows; carries the 1-based line number."""

    def __init__(self, lineno: int, message: str) -> None:
        super().__init__(f"line {lineno}: {message}")
        self.lineno = lineno


def parse_bool(text: str, default: bool = True) -> bool:
    word = text.strip().lower()
    if not word:
        return default
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"not a boolean: {text!r}")


def parse_stock(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    return int(text)


def parse_row(fields: List[str], lineno: int = 0) -> Item:
    """Turn one CSV row (already split into fields) into an Item."""
    if len(fields) < 3:
        raise CatalogError(lineno, f"expected at least 3 fields, got {len(fields)}")
    fields = fields + [""] * (len(HEADER) - len(fields))
    try:
        price = float(fields[COL_PRICE])
        stock = parse_stock(fields[COL_STOCK])
        taxable = parse_bool(fields[COL_TAXABLE])
    except ValueError as exc:
        raise CatalogError(lineno, str(exc)) from exc
    return Item(
        sku=fields[COL_SKU].strip(),
        name=fields[COL_NAME].strip(),
        unit_price=price,
        taxable=taxable,
        stock=stock,
    )


def load_catalog(text: str) -> Dict[str, Item]:
    """Parse catalogue CSV text into a dict keyed by sku."""
    reader = csv.reader(io.StringIO(text))
    items: Dict[str, Item] = {}
    for lineno, fields in enumerate(reader, start=1):
        if lineno == 1 and [f.strip().lower() for f in fields[:3]] == HEADER[:3]:
            continue
        if not fields or not "".join(fields).strip():
            continue
        item = parse_row(fields, lineno)
        if item.sku in items:
            raise CatalogError(lineno, f"duplicate sku {item.sku}")
        items[item.sku] = item
    return items


def dump_catalog(items: Iterable[Item]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(HEADER)
    for item in sorted(items, key=lambda i: i.sku):
        stock = "" if item.stock is None else str(item.stock)
        writer.writerow([item.sku, item.name, f"{item.unit_price:.2f}", stock, "yes" if item.taxable else "no"])
    return out.getvalue()
