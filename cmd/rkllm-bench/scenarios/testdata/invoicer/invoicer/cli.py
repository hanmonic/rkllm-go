"""Command line entry point: `invoicer render ORDER.json --catalog items.csv`."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from . import report, storage
from .models import Customer, Invoice


def build_invoice(order: dict, catalog: dict) -> Invoice:
    cust = order["customer"]
    customer = Customer(
        customer_id=cust["id"],
        name=cust["name"],
        email=cust.get("email", ""),
        tax_exempt=bool(cust.get("tax_exempt", False)),
        region=cust.get("region", "default"),
    )
    invoice = Invoice(
        number=order["number"],
        customer=customer,
        issued=date.fromisoformat(order.get("issued", date.today().isoformat())),
        coupon=order.get("coupon"),
    )
    for line in order["lines"]:
        item = catalog.get(line["sku"])
        if item is None:
            raise SystemExit(f"unknown sku {line['sku']}")
        if not item.in_stock(line["qty"]):
            print(f"warning: {item.sku} has only {item.stock} in stock", file=sys.stderr)
        invoice.add(item, int(line["qty"]), line.get("note", ""))
    return invoice


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="invoicer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render an order as a text invoice")
    r.add_argument("order")
    r.add_argument("--catalog", required=True)
    s = sub.add_parser("summary", help="print the one-line summary")
    s.add_argument("order")
    s.add_argument("--catalog", required=True)
    args = ap.parse_args(argv)

    with open(args.catalog, encoding="utf-8") as fh:
        catalog = storage.load_catalog(fh.read())
    with open(args.order, encoding="utf-8") as fh:
        order = json.load(fh)
    invoice = build_invoice(order, catalog)
    if args.cmd == "render":
        sys.stdout.write(report.render(invoice))
    else:
        print(report.summary_line(invoice))
    return 0


if __name__ == "__main__":
    sys.exit(main())
