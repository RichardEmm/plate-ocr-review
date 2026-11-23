import asyncio
import json

import pytest

from plate_ocr_review.core import HEADERS, Report, Result, Settings
from plate_ocr_review.sheets import ExportConflict, export_report, recover_creation, spreadsheet_id


class FakeSheets:
    def __init__(self):
        self.calls = []
        self.values = {}
        self.tabs = []
        self.id = "1234567890123456789012345"
        self.fail_write = False
        self.fail_create = False
        self.break_readback = False

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "sheets_create":
            self.tabs.append({"title": args["sheetTitle"], "sheetId": 42})
            if self.fail_create:
                raise TimeoutError("Ambiguous creation timeout")
            return {"spreadsheetId": self.id}
        if name == "sheets_getMetadata":
            return {"sheets": self.tabs}
        if name == "sheets_batchUpdate":
            if "addSheet" in args["requests"][0]:
                tab = args["requests"][0]["addSheet"]["properties"] | {"sheetId": 43}
                self.tabs.append(tab)
                return {"replies": [{"addSheet": {"properties": tab}}]}
            return {}
        if name == "sheets_getRange":
            rows = self.values.get(args["range"], [])
            if self.break_readback and rows:
                return {"values": [["WRONG"]]}
            return {"values": rows}
        if name == "sheets_updateRange":
            self.values[args["range"]] = args["values"]
            if self.fail_write:
                self.fail_write = False
                raise TimeoutError("Ambiguous write timeout")
            return {}
        raise AssertionError(name)


@pytest.fixture
def report(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    report = Report("images", Settings(), complete=True, discovered=1)
    result = Result("image.jpg", "2026-09-08 17:06:27", "DOVER24", "00123", "00123", [.99] * 5)
    result.classify(.9)
    report.results.append(result)
    report.save()
    return report


def run(report, fake, target=""):
    return asyncio.run(export_report(report, fake, "ANPR test", target))


def test_export_exact_columns_raw_and_readback(report):
    fake = FakeSheets()
    url = run(report, fake)
    assert fake.id in url
    write = next(args for name, args in fake.calls if name == "sheets_updateRange")
    assert write["valueInputOption"] == "RAW"
    assert write["values"] == [HEADERS, report.results[0].sheet_row()]
    assert all(len(row) == 5 for row in write["values"])
    assert json.loads((report.directory / "sheet-receipt.json").read_text())["state"] == "verified"


def test_retry_no_new_sheet_or_duplicate_rows(report):
    fake = FakeSheets()
    fake.fail_write = True
    with pytest.raises(TimeoutError):
        run(report, fake)
    run(report, fake)
    assert sum(name == "sheets_create" for name, _ in fake.calls) == 1
    assert sum(name == "sheets_updateRange" for name, _ in fake.calls) == 1


def test_existing_sheet_new_tab_only(report):
    fake = FakeSheets()
    fake.tabs = [{"title": "Important existing data", "sheetId": 1}]
    run(report, fake, fake.id)
    assert fake.tabs[0]["title"] == "Important existing data"
    assert not any(name == "sheets_create" for name, _ in fake.calls)
    assert all("Important existing data" not in args.get("range", "") for _, args in fake.calls)


def test_external_edits_not_overwritten(report):
    fake = FakeSheets()
    run(report, fake)
    a1 = next(iter(fake.values))
    fake.values[a1] = [HEADERS, ["2026-09-08 17:06:27", "DOVER24", "edited", "00123", "SAME"]]
    before = sum(name == "sheets_updateRange" for name, _ in fake.calls)
    with pytest.raises(ExportConflict, match="edited externally"):
        run(report, fake)
    assert sum(name == "sheets_updateRange" for name, _ in fake.calls) == before


def test_manual_review_reexport_updates_original_tab(report):
    fake = FakeSheets()
    run(report, fake)
    report.results[0].manual_crop = True
    report.results[0].classify(.9)
    run(report, fake)
    assert next(iter(fake.values.values()))[1][-1] == "SAME"
    assert len(fake.tabs) == 1


def test_clipped_status_export_and_reexport(report):
    fake = FakeSheets()
    run(report, fake)
    result = report.results[0]
    result.ocr_plate = "0012"
    result.confidences = [.99] * 4
    result.manual_crop = True
    result.classify(.90)
    assert result.status == "CLIPPED"
    run(report, fake)
    rows = next(iter(fake.values.values()))
    assert rows[1][-1] == "CLIPPED"
    assert all(len(row) == 5 for row in rows)
    assert len(fake.tabs) == 1


def test_readback_failure_not_success(report):
    fake = FakeSheets()
    fake.break_readback = True
    with pytest.raises(RuntimeError, match="read-back failed"):
        run(report, fake)
    assert json.loads((report.directory / "sheet-receipt.json").read_text())["state"] != "verified"


def test_ambiguous_creation_no_blind_duplicate_and_recover(report):
    fake = FakeSheets()
    fake.fail_create = True
    with pytest.raises(TimeoutError):
        run(report, fake)
    with pytest.raises(ExportConflict, match="creation was interrupted"):
        run(report, fake)
    assert len(fake.tabs) == 1
    asyncio.run(recover_creation(report, fake, fake.id))
    run(report, fake)
    assert len(fake.tabs) == 1


def test_change_destination_rejected(report):
    fake = FakeSheets()
    run(report, fake)
    with pytest.raises(ExportConflict, match="already has a destination"):
        run(report, fake, "different12345678901234567890")


def test_large_report_batches(report):
    report.results *= 1001
    fake = FakeSheets()
    run(report, fake)
    batches = [args["values"] for name, args in fake.calls if name == "sheets_updateRange"]
    assert [len(b) for b in batches] == [500, 500, 2]


def test_sheet_url_parsing():
    expected = "1234567890123456789012345"
    assert spreadsheet_id(f"https://docs.google.com/spreadsheets/d/{expected}/edit#gid=0") == expected
    assert spreadsheet_id(expected) == expected
    assert spreadsheet_id("") == ""
    with pytest.raises(ValueError):
        spreadsheet_id("https://evil.example/sheet")


def test_flat_metadata_never_shrinks_existing_grid(report):
    fake = FakeSheets()
    run(report, fake)
    fake.tabs[0]["rowCount"] = 50000
    fake.tabs[0]["columnCount"] = 5
    fake.calls.clear()
    run(report, fake)
    resize = next(args["requests"][0]["updateSheetProperties"] for name, args in fake.calls
                  if name == "sheets_batchUpdate" and "updateSheetProperties" in args["requests"][0])
    assert resize["properties"]["gridProperties"]["rowCount"] == 50000
