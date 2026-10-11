import pytest

from invoicer.storage import CatalogError, dump_catalog, load_catalog, parse_bool, parse_row

CATALOG = """sku,name,unit_price,stock,taxable
W-100,Widget,10.00,250,yes
G-200,Gadget,4.50,,no
S-300,Sprocket,0.75,1200,
"""


def test_parse_bool():
    assert parse_bool("yes") is True
    assert parse_bool(" No ") is False
    assert parse_bool("", default=False) is False
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_load_catalog_basic():
    items = load_catalog(CATALOG)
    assert sorted(items) == ["G-200", "S-300", "W-100"]
    widget = items["W-100"]
    assert widget.unit_price == 10.0
    assert widget.stock == 250
    assert widget.taxable is True


def test_load_catalog_untracked_stock():
    items = load_catalog(CATALOG)
    assert items["G-200"].stock is None
    assert items["G-200"].taxable is False
    assert items["S-300"].taxable is True


def test_load_catalog_rejects_duplicate_sku():
    with pytest.raises(CatalogError, match="duplicate sku W-100"):
        load_catalog(CATALOG + "W-100,Widget again,11.00,5,yes\n")


def test_dump_roundtrip():
    items = load_catalog(CATALOG)
    again = load_catalog(dump_catalog(items.values()))
    assert again == items


def test_short_row_rejected():
    with pytest.raises(CatalogError, match="line 7"):
        parse_row(["X-1", "Thing"], lineno=7)
