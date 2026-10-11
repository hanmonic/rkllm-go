from datetime import date

import pytest

from invoicer.models import Customer, Invoice, Item
from invoicer.report import render, summary_line

WIDGET = Item(sku="W-100", name="Widget", unit_price=10.0)
ADA = Customer(customer_id="C-1", name="Ada Lovelace", email="ada@example.com")


def make_invoice(qty):
    inv = Invoice(number="2024-0042", customer=ADA, issued=date(2024, 3, 1))
    inv.add(WIDGET, qty)
    return inv


def test_render_header():
    text = render(make_invoice(2))
    first = text.splitlines()[0]
    assert first.startswith("INVOICE 2024-0042")
    assert first.endswith("issued 2024-03-01")
    assert "Bill to: Ada Lovelace <ada@example.com>" in text


def test_render_bulk_line():
    text = render(make_invoice(10))
    line = [l for l in text.splitlines() if l.startswith("W-100")][0]
    assert line.split()[-1] == "85.00"


def test_summary_line():
    assert summary_line(make_invoice(2)) == "2024-0042 C-1 units=2 total=21.60"


def test_empty_invoice_refused():
    inv = Invoice(number="2024-0043", customer=ADA, issued=date(2024, 3, 1))
    with pytest.raises(ValueError):
        render(inv)
