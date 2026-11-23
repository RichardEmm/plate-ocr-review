"""Resumable Sheets export: exact five columns, no edits to unrelated tabs."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .core import HEADERS, Report, atomic_json, data_directory

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class ExportConflict(RuntimeError):
    pass


@asynccontextmanager
async def workspace_connection(command: str):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    executable = Path(command).expanduser()
    if not executable.is_file():
        raise FileNotFoundError(
            "Google Workspace connector not found. Choose OAuth sign-in or configure its launcher."
        )
    env = dict(os.environ)
    env["WORKSPACE_FEATURE_OVERRIDES"] = (
        "drive.read:on,drive.write:on,sheets.read:on,sheets.write:on,"
        "gmail.read:off,gmail.write:off,docs.read:off,docs.write:off,"
        "calendar.read:off,calendar.write:off,chat.read:off,chat.write:off,"
        "people.read:off,slides.read:off,slides.write:off,time.read:off,"
        "tasks.read:off,tasks.write:off"
    )
    async with stdio_client(StdioServerParameters(command=str(executable), env=env)) as streams:
        async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=120)) as session:
            await session.initialize()

            async def call(name: str, args: dict) -> dict:
                response = await session.call_tool(name, args)
                text = "\n".join(item.text for item in response.content if item.type == "text")
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Unexpected Google Workspace response for {name}") from exc
                if response.isError or value.get("error"):
                    raise RuntimeError(f"Google Workspace {name}: {value.get('error', text)}")
                return value

            yield call


def oauth_session(client_file: str):
    """Independent OAuth option; never read or copy the connector's credentials."""
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token = data_directory() / "oauth-token.json"
    credentials = Credentials.from_authorized_user_file(str(token), SCOPES) if token.exists() else None
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials or not credentials.valid:
        config = Path(client_file).expanduser()
        if not config.is_file():
            raise ValueError("Select a Google OAuth Desktop app client JSON file")
        flow = InstalledAppFlow.from_client_secrets_file(str(config), SCOPES)
        credentials = flow.run_local_server(port=0, timeout_seconds=180)
    token.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(token, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(credentials.to_json())
    return AuthorizedSession(credentials)


def oauth_call(session, name: str, args: dict) -> dict:
    """Map the same narrow tool contract to Google's REST API."""
    from urllib.parse import quote

    base = "https://sheets.googleapis.com/v4/spreadsheets"
    sheet_id = args.get("spreadsheetId", "")
    if name == "sheets_create":
        response = session.post(base, json={
            "properties": {"title": args["title"]},
            "sheets": [{"properties": {"title": args["sheetTitle"]}}],
        }, timeout=60)
    elif name == "sheets_getMetadata":
        response = session.get(f"{base}/{sheet_id}", timeout=60)
    elif name == "sheets_batchUpdate":
        response = session.post(f"{base}/{sheet_id}:batchUpdate", json={
            "requests": args["requests"]}, timeout=60)
    else:
        url = f"{base}/{sheet_id}/values/{quote(args['range'], safe='')}"
        if name == "sheets_getRange":
            response = session.get(url, timeout=60)
        elif name == "sheets_updateRange":
            response = session.put(url, params={"valueInputOption": "RAW"}, json={
                "range": args["range"], "values": args["values"]}, timeout=60)
        else:
            raise ValueError("Unsupported Google operation")
    response.raise_for_status()
    data = response.json()
    if name == "sheets_getMetadata":
        data["sheets"] = [sheet["properties"] for sheet in data["sheets"]]
    return data


def spreadsheet_id(value: str) -> str:
    import re

    value = value.strip()
    if not value:
        return ""
    match = re.fullmatch(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)(?:/[^\s]*)?", value)
    if match:
        return match[1]
    if re.fullmatch(r"[\w-]{20,}", value):
        return value
    raise ValueError("Enter a Google Sheets URL or spreadsheet ID, or leave it blank")


def normalize_metadata(value: dict) -> list[dict]:
    tabs = []
    for sheet in value["sheets"]:
        tab = dict(sheet.get("properties", sheet))
        if "gridProperties" not in tab and "rowCount" in tab:
            tab["gridProperties"] = {"rowCount": tab["rowCount"],
                                     "columnCount": tab.get("columnCount", 5)}
        tabs.append(tab)
    return tabs


def normalized_rows(values: list, count: int) -> list[list[str]]:
    return [[str(row[c]) if c < len(row) and row[c] is not None else "" for c in range(5)]
            for row in (values + [[]] * max(0, count - len(values)))[:count]]


async def export_report(
    report: Report,
    call,
    title: str,
    target: str = "",
    progress: Callable[[str], None] = lambda _: None,
) -> str:
    if not report.results:
        raise ValueError("There are no results to export")
    receipt_path = report.directory / "sheet-receipt.json"
    target_id = spreadsheet_id(target)
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt and target_id and receipt["spreadsheetId"] != target_id:
        raise ExportConflict("This run already has a destination. Start a new run for another sheet.")

    # Record creation intent *before* submitting; a timed-out create must be
    # resolved by inspecting the intended title, not blindly creating duplicates.
    if receipt is None:
        tab = "OCR " + report.run_id
        receipt = {"spreadsheetId": target_id, "tab": tab, "state": "creating",
                   "title": title.strip() or "ANPR OCR " + datetime.now().isoformat(" ", "minutes")}
        atomic_json(receipt_path, receipt)
        if target_id:
            added = await call("sheets_batchUpdate", {
                "spreadsheetId": target_id,
                "requests": [{"addSheet": {"properties": {"title": tab, "gridProperties": {
                    "rowCount": max(1000, len(report.results) + 1), "columnCount": 5}}}}],
            })
            receipt["sheetId"] = added["replies"][0]["addSheet"]["properties"]["sheetId"]
        else:
            added = await call("sheets_create", {"title": receipt["title"], "sheetTitle": tab})
            receipt["spreadsheetId"] = added["spreadsheetId"]
        receipt["state"] = "writing"
        atomic_json(receipt_path, receipt)
    elif receipt["state"] == "creating":
        if receipt["spreadsheetId"]:
            meta = await call("sheets_getMetadata", {"spreadsheetId": receipt["spreadsheetId"]})
            tabs = [s for s in normalize_metadata(meta) if s["title"] == receipt["tab"]]
            if len(tabs) == 1:
                receipt["sheetId"] = tabs[0]["sheetId"]
                receipt["state"] = "writing"
                atomic_json(receipt_path, receipt)
            else:
                raise ExportConflict("Creation was interrupted. Inspect the destination before retrying.")
        else:
            raise ExportConflict(
                f"Sheet creation was interrupted. Find the Google Sheet titled {receipt['title']!r}, "
                "paste its URL in the destination field, and use Recover export. "
                "No automatic duplicate was created."
            )
    sheet_id = receipt["spreadsheetId"]
    meta = await call("sheets_getMetadata", {"spreadsheetId": sheet_id})
    tabs = [s for s in normalize_metadata(meta) if s["title"] == receipt["tab"]]
    if len(tabs) != 1 or ("sheetId" in receipt and tabs[0]["sheetId"] != receipt["sheetId"]):
        raise ExportConflict("The report tab was changed or removed; it was not overwritten")
    receipt["sheetId"] = tabs[0]["sheetId"]
    values = [HEADERS] + [r.sheet_row() for r in report.results]
    previous = receipt.get("rows", [])
    if len(previous) > len(values):
        raise ExportConflict("This report is shorter than its existing export; start a new run")
    receipt["state"] = "writing"
    atomic_json(receipt_path, receipt)
    await call("sheets_batchUpdate", {"spreadsheetId": sheet_id, "requests": [
        {"updateSheetProperties": {"properties": {"sheetId": receipt["sheetId"],
            "gridProperties": {"rowCount": max(tabs[0].get("gridProperties", {}).get("rowCount", 1000),
                                                   len(values)), "frozenRowCount": 1}},
            "fields": "gridProperties.rowCount,gridProperties.frozenRowCount"}}]})
    for offset in range(0, len(values), 500):
        batch = values[offset:offset+500]
        a1 = f"'{receipt['tab']}'!A{offset+1}:E{offset+len(batch)}"
        current = await call("sheets_getRange", {"spreadsheetId": sheet_id, "range": a1})
        live = normalized_rows(current.get("values", []), len(batch))
        old = previous[offset:offset+len(batch)]
        for i, row in enumerate(live):
            if any(row) and row != batch[i] and not (i < len(old) and row == old[i]):
                raise ExportConflict(f"Row {offset+i+1} was edited externally; it was not overwritten")
        # Exact desired values are always allowed when retrying an interrupted
        # write; the receipt retains the preceding verified version until read-back.
        if live != batch:
            await call("sheets_updateRange", {"spreadsheetId": sheet_id, "range": a1,
                                              "values": batch, "valueInputOption": "RAW"})
        back = await call("sheets_getRange", {"spreadsheetId": sheet_id, "range": a1})
        if normalized_rows(back.get("values", []), len(batch)) != batch:
            raise RuntimeError(f"Google Sheets read-back failed at row {offset+1}")
        progress(f"Verified {min(offset+len(batch), len(values))-1:,} / {len(values)-1:,} images")
        # Keep a rolling verified receipt. It remains safe to retry a write whose
        # read-back failed because exact desired values are always allowed.
        updated = receipt.get("rows", previous.copy())
        updated[offset:offset+len(batch)] = batch
        receipt["rows"] = updated
        atomic_json(receipt_path, receipt)
    await call("sheets_batchUpdate", {"spreadsheetId": sheet_id, "requests": formatting_requests(
        receipt["sheetId"], len(values))})
    receipt["state"] = "verified"
    receipt["url"] = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit#gid={receipt['sheetId']}"
    atomic_json(receipt_path, receipt)
    return receipt["url"]


def formatting_requests(sheet_id: int, rows: int) -> list[dict]:
    entire = {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": rows,
              "startColumnIndex": 0, "endColumnIndex": 5}
    requests = [
        {"setBasicFilter": {"filter": {"range": entire}}},
        {"repeatCell": {"range": {**entire, "endRowIndex": 1}, "cell": {
            "userEnteredFormat": {"backgroundColor": {"red": .08, "green": .16, "blue": .23},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}}},
            "fields": "userEnteredFormat"}},
        {"autoResizeDimensions": {"dimensions": {"sheetId": sheet_id, "dimension": "COLUMNS",
                                                  "startIndex": 0, "endIndex": 5}}},
    ]
    # Idempotent formatting avoids accumulating conditional format rules on
    # every retry/review export. All status text remains literal.
    requests.append({"repeatCell": {"range": {**entire, "startRowIndex": 1, "startColumnIndex": 4},
        "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
        "fields": "userEnteredFormat.textFormat.bold"}})
    return requests


async def recover_creation(report: Report, call, target: str) -> None:
    path = report.directory / "sheet-receipt.json"
    receipt = json.loads(path.read_text())
    if receipt["state"] != "creating" or receipt["spreadsheetId"]:
        raise ValueError("There is no unresolved spreadsheet creation for this report")
    destination = spreadsheet_id(target)
    if not destination:
        raise ValueError("Paste the URL of the previously created Google Sheet")
    meta = await call("sheets_getMetadata", {"spreadsheetId": destination})
    tabs = [s for s in normalize_metadata(meta) if s["title"] == receipt["tab"]]
    if len(tabs) != 1:
        raise ExportConflict("This sheet does not contain this run's unique tab")
    receipt.update(spreadsheetId=destination, sheetId=tabs[0]["sheetId"], state="writing")
    atomic_json(path, receipt)


def run_export(report: Report, backend: str, connector: str, client_file: str,
               title: str, target: str, progress=lambda _: None, recover=False) -> str:
    async def work(call):
        async def throttled(name, args):
            for attempt in range(4):
                try:
                    return await call(name, args)
                except Exception as exc:
                    # Retry only reads and idempotent writes on rate limits, never
                    # creation or addSheet requests (ambiguous write outcomes).
                    creation = name == "sheets_create" or (
                        name == "sheets_batchUpdate" and any("addSheet" in r for r in args["requests"]))
                    if creation or attempt == 3 or not any(
                            word in str(exc).lower() for word in ["429", "quota", "rate limit"]):
                        raise
                    progress("Google quota reached; retrying in 65 seconds…")
                    await asyncio.sleep(65)
        if recover:
            await recover_creation(report, throttled, target)
        return await export_report(report, throttled, title, target, progress)

    if backend == "Workspace connector":
        async def connected():
            async with workspace_connection(connector) as call:
                return await work(call)
        return asyncio.run(connected())
    session = oauth_session(client_file)
    try:
        async def call(name, args):
            return oauth_call(session, name, args)
        return asyncio.run(work(call))
    finally:
        session.close()
