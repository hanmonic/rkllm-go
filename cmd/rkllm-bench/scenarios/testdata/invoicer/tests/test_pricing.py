from datetime import date

import pytest

from invoicer.models import Customer, Invoice, Item
from invoicer.pricing import breakdown, line_total, round_money, total

WIDGET = Item(sku="W-100", name="Widget", unit_price=10.0)
GADGET = Item(sku="G-200", name="Gadget", unit_price=4.5, taxable=False)
ADA = Customer(customer_id="C-1", name="Ada Lovelace", email="ada@example.com")


def make_invoice(*lines, coupon=None, customer=ADA):
    inv = Invoice(number="2024-0042", customer=customer, issued=date(2024, 3, 1), coupon=coupon)
    for item, qty in lines:
        inv.add(item, qty)
    return inv


def test_line_total_basic():
    assert line_total(WIDGET, 3) == 30.0


def test_bulk_discount_below_threshold():
    assert line_total(WIDGET, 9) == 90.0


def test_bulk_discount_applies_at_threshold():
    # 10 units is "10 or more": the discount must apply.
    assert line_total(WIDGET, 10) == 85.0


def test_bulk_discount_above_threshold():
    assert line_total(WIDGET, 12) == 102.0


def test_coupon_applied_after_bulk_discount():
    inv = make_invoice((WIDGET, 12), (GADGET, 2), coupon="welcome10")
    figures = breakdown(inv)
    assert figures["subtotal"] == 111.0
    assert figures["discount"] == 11.1
    # only the widget line is taxable: 102.00 * 0.9 * 0.08
    assert figures["tax"] == 7.34
    assert figures["total"] == 107.24


def test_tax_exempt_customer_pays_no_tax():
    exempt = Customer(customer_id="C-2", name="Charity", tax_exempt=True)
    inv = make_invoice((WIDGET, 3), customer=exempt)
    assert breakdown(inv)["tax"] == 0.0
    assert total(inv) == 30.0


@pytest.mark.parametrize("value,expected", [(0.125, 0.13), (2.675, 2.68), (-1.125, -1.13), (3.0, 3.0)])
def test_round_money_half_away_from_zero(value, expected):
    assert round_money(value) == expected
