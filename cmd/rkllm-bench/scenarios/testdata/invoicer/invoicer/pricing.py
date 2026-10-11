"""Pricing rules: bulk discounts, coupons and sales tax.

The rules, in the order they are applied to an invoice:

1. Bulk discount, per line: ordering BULK_THRESHOLD units or more of the
   same item takes BULK_DISCOUNT off that line.
2. Coupon, per invoice: a known coupon code takes a percentage off the
   discounted subtotal. Unknown codes are ignored (and logged).
3. Sales tax, per line: taxable lines are taxed at the customer's regional
   rate, unless the customer is tax exempt.

All intermediate values are floats; round_money is applied to each line
total and to the final figures only.
"""
from __future__ import annotations

import logging
from typing import Dict

from .models import Customer, Invoice, Item, LineItem

log = logging.getLogger(__name__)

BULK_THRESHOLD = 10
BULK_DISCOUNT = 0.15

COUPONS: Dict[str, float] = {
    "WELCOME10": 0.10,
    "SPRING20": 0.20,
    "STAFF": 0.30,
}

TAX_RATES: Dict[str, float] = {
    "default": 0.08,
    "north": 0.06,
    "south": 0.0725,
    "export": 0.0,
}


def round_money(value: float) -> float:
    """Round to cents, half away from zero (what the accounting team expects)."""
    cents = abs(value) * 100 + 0.5
    rounded = int(cents) / 100
    return rounded if value >= 0 else -rounded


def line_total(item: Item, qty: int) -> float:
    """Total for qty units of item, applying the bulk discount for orders
    of BULK_THRESHOLD units or more. Tax is not included."""
    subtotal = item.unit_price * qty
    if qty >= BULK_THRESHOLD:
        subtotal = subtotal * (1 - BULK_DISCOUNT)
    return round_money(subtotal)


def coupon_rate(code: str) -> float:
    """Discount rate for a coupon code; 0.0 for unknown or empty codes."""
    if not code:
        return 0.0
    rate = COUPONS.get(code.strip().upper())
    if rate is None:
        log.warning("ignoring unknown coupon %r", code)
        return 0.0
    return rate


def tax_rate(customer: Customer) -> float:
    if customer.tax_exempt:
        return 0.0
    return TAX_RATES.get(customer.region, TAX_RATES["default"])


def subtotal(invoice: Invoice) -> float:
    """Sum of line totals after bulk discounts, before coupon and tax."""
    return round_money(sum(line_total(l.item, l.qty) for l in invoice.lines))


def discount(invoice: Invoice) -> float:
    """Coupon discount amount for the invoice (a positive number)."""
    return round_money(subtotal(invoice) * coupon_rate(invoice.coupon or ""))


def line_tax(line: LineItem, customer: Customer, coupon: float = 0.0) -> float:
    if not line.item.taxable:
        return 0.0
    base = line_total(line.item, line.qty) * (1 - coupon)
    return base * tax_rate(customer)


def tax(invoice: Invoice) -> float:
    rate = coupon_rate(invoice.coupon or "")
    return round_money(sum(line_tax(l, invoice.customer, rate) for l in invoice.lines))


def total(invoice: Invoice) -> float:
    """Amount due: subtotal - coupon discount + tax."""
    return round_money(subtotal(invoice) - discount(invoice) + tax(invoice))


def breakdown(invoice: Invoice) -> Dict[str, float]:
    """All the figures printed at the bottom of an invoice."""
    return {
        "subtotal": subtotal(invoice),
        "discount": discount(invoice),
        "tax": tax(invoice),
        "total": total(invoice),
    }
