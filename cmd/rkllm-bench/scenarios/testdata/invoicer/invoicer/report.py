"""Plain-text invoice rendering.

The layout is fixed-width so invoices can be printed on the receipt printer
in the shop as well as attached to emails:

    INVOICE 2024-0042                       issued 2024-03-01
    Bill to: Ada Lovelace <ada@example.com>
    ------------------------------------------------------------
    SKU      Item                     Qty     Unit      Total
    W-100    Widget                    12    10.00     102.00
    ------------------------------------------------------------
                                     Subtotal          102.00
                                     Discount           -0.00
                                     Tax                 8.16
                                     TOTAL             110.16
"""
from __future__ import annotations

from typing import List

from . import pricing
from .models import Invoice

WIDTH = 60
NAME_WIDTH = 24


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _money(value: float) -> str:
    return f"{value:10.2f}"


def header(invoice: Invoice) -> List[str]:
    left = f"INVOICE {invoice.number}"
    right = f"issued {invoice.issued.isoformat()}"
    lines = [left + right.rjust(WIDTH - len(left))]
    bill_to = invoice.customer.name
    if invoice.customer.email:
        bill_to += f" <{invoice.customer.email}>"
    lines.append(f"Bill to: {bill_to}")
    if invoice.customer.tax_exempt:
        lines.append("Tax exempt customer")
    return lines


def body(invoice: Invoice) -> List[str]:
    lines = [_rule(), f"{'SKU':<8} {'Item':<{NAME_WIDTH}} {'Qty':>4} {'Unit':>8} {'Total':>10}"]
    for line in invoice.lines:
        name = line.item.name
        if len(name) > NAME_WIDTH:
            name = name[: NAME_WIDTH - 1] + "~"
        amount = pricing.line_total(line.item, line.qty)
        lines.append(f"{line.item.sku:<8} {name:<{NAME_WIDTH}} {line.qty:>4} {line.item.unit_price:>8.2f} {amount:>10.2f}")
        if line.note:
            lines.append(f"         note: {line.note}")
    lines.append(_rule())
    return lines


def footer(invoice: Invoice) -> List[str]:
    figures = pricing.breakdown(invoice)
    pad = " " * 33
    lines = [
        f"{pad}{'Subtotal':<12}{_money(figures['subtotal'])}",
        f"{pad}{'Discount':<12}{_money(-figures['discount'])}",
        f"{pad}{'Tax':<12}{_money(figures['tax'])}",
        f"{pad}{'TOTAL':<12}{_money(figures['total'])}",
    ]
    if invoice.coupon:
        lines.insert(1, f"{pad}coupon {invoice.coupon.upper()}")
    return lines


def render(invoice: Invoice) -> str:
    """Render the whole invoice as text, one trailing newline."""
    if invoice.is_empty():
        raise ValueError("refusing to render an empty invoice")
    return "\n".join(header(invoice) + body(invoice) + footer(invoice)) + "\n"


def summary_line(invoice: Invoice) -> str:
    """One-line summary used in the daily sales log."""
    return f"{invoice.number} {invoice.customer.customer_id} units={invoice.units()} total={pricing.total(invoice):.2f}"
