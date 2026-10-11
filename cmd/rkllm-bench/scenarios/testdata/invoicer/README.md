# invoicer

Small invoicing library shared by the shop till and the web store.

## Layout

- `invoicer/models.py`: dataclasses (Item, LineItem, Customer, Invoice)
- `invoicer/pricing.py`: bulk discount, coupons, sales tax, totals
- `invoicer/storage.py`: catalogue CSV load/dump
- `invoicer/report.py`: fixed-width text rendering
- `invoicer/cli.py`: `invoicer render ORDER.json --catalog items.csv`
- `tests/`: pytest suite

## Pricing rules

1. Bulk discount: 10 or more units of the same item on one line get 15% off that line.
2. Coupons: WELCOME10, SPRING20, STAFF take 10/20/30% off the discounted subtotal.
3. Tax: regional rate per taxable line, applied after the coupon; tax-exempt customers pay none.

## Development

    pip install -e .[test]
    pytest -q

The tests are the spec: do not change a test to make it pass, fix the code.
