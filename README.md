# Plate OCR Review

A local desktop application for comparing ANPR camera readings with
[fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr), recursively across
an image folder, and recording the results in Google Sheets.

## Quick start

```bash
git clone https://github.com/RichardEmm/plate-ocr-review.git
cd plate-ocr-review
python -m venv .venv
.venv/bin/python -m pip install -e .
./run.sh
```

Python 3.11+ and a graphical desktop are required. See the installation section
below for Windows and standalone Google sign-in.

1. Click **Choose folder…**. All subfolders are included.
2. Set **Filename configuration** and check the **Test filename** preview.
3. Set the confidence threshold if needed (default **90%**).
4. Click **Run OCR**. First use downloads the selected global ONNX model.
5. Filter by status and preview any image. Use **Clipped / overly cropped (manual flag)**
   for clipping missed by the automatic heuristic.
6. Click **Export to Google Sheets**, then **Open Sheet**.

The default export backend expects an existing Google Workspace MCP connector,
with its launcher path configured in the app. That connector is not included in
this repository. It uses the connector's existing sign-in without copying
credentials. If you have the matching local login helper, renew sign-in with
`~/.local/bin/google-workspace-login` in a local terminal. Otherwise choose
**Google OAuth** and follow the standalone sign-in instructions below.
Do not paste tokens or passwords into the application or chat.

No folder has been processed automatically and no production Google Sheet is
created merely by starting the app.

## Results: exactly five columns

| Date/Time | Camera | Camera interpretation | fast-plate-ocr interpretation | Status |
|---|---|---|---|---|
| 2026-09-08 17:06:27 | DOVER24 | GSV81G | GSV81G | SAME |

For `26-09-08_17-06-27_DOVER24_GSV81G_ABC.jpg`, the date is interpreted as
**YY-MM-DD**, the time as **HH-MM-SS**, the camera is `DOVER24`, and `GSV81G` is
the camera reading with the default configuration. Camera names do not need
to begin with `DOVER`; any non-empty camera name is accepted and its case is
preserved. Unselected fields, including suffixes, are ignored.

### Configure filename fields

Under **Filename configuration**, set the literal **Separator** (default `_`)
and the field positions for **Camera**, **Plate**, **Date**, and **Time**.
Positions count from **1**, excluding the file extension. Negative positions
count from the end: `-1` is the last field, `-2` is the second-to-last.
Zero, duplicate positions, missing fields and empty selected fields are invalid.
The date/time formats remain `YY-MM-DD` and `HH-MM-SS`.

For `26-09-08_17-06-27_DOVER24_GSV81G_ABC.jpg`:

| Field | Position | Value |
|---|---:|---|
| Date | 1 | 26-09-08 |
| Time | 2 | 17-06-27 |
| Camera | 3 | DOVER24 |
| Plate | 4 (or -2) | GSV81G |

For `26-09-08_17-06-27_GSV81G_GateEast_ABC.jpg`, use **Camera = 4** and
**Plate = 3**. With extra prefixes but a consistent final suffix, relative
positions such as **Camera = -3** and **Plate = -2** are useful. When mixing
files with and without suffixes, use stable positive positions rather than
relative positions that shift. A camera name or plate must fit in one field;
choose a separator that is not part of those values.

Type a sample filename or click **Choose example…** to see the numbered fields
and extracted date/time, camera and plate before running OCR. This only inspects
the filename; it does not run OCR or upload anything. Configuration changes apply
to **new OCR runs**, not existing results or existing Google Sheet rows.

The configuration is remembered automatically and is captured in each run's
`report.json` for repeatability. It can also be edited in the `filename` object
of `~/.local/share/plate-ocr-review/preferences.json` while the app is closed:

```json
"filename": {
  "separator": "_",
  "camera_field": 3,
  "plate_field": 4,
  "date_field": 1,
  "time_field": 2
}
```

Merge this object into the existing preferences JSON; do not replace the other
preferences. Older saved reports without this object remain readable.

- **SAME (highest priority):** the non-empty OCR text is exactly the same as the
  extracted camera interpretation. This applies regardless of confidence, crop
  flags, manual clipping flags or filename/processing warnings. Two empty values
  do not count as a plate match. Confidence and warnings remain visible for review.
- **CLIPPED:** clipping is detected automatically at an image edge, or manually
  flagged, **and the OCR and camera readings differ**, **and either** at least
  one recognised character has confidence **below 60%** **or** the OCR string
  length differs from the camera string length. This rule takes precedence over
  `LOWCONFIDENCE` and uses a fixed 60% cutoff, independent of the configurable
  high-confidence threshold. Exactly 60% does not meet the confidence condition.
- **SAME (normalized agreement):** readings match after ignoring case and
  separators, every recognised character is at or above the threshold, and no
  crop/error flags apply. This preserves the earlier comparison behavior; only
  an exact text match bypasses confidence and warnings.
- **DIFF:** every recognised character is at or above the threshold, no crop/error
  flags apply, and the readings differ.
- **LOWCONFIDENCE:** a character is below the threshold, confidence is absent
  or invalid, the image is too small or possibly clipped, a manual crop flag
  is set, or the filename/image cannot be read reliably, and neither the exact-match
  `SAME` override nor the `CLIPPED` rule is met. Clipped images with exactly
  matching readings are labelled `SAME`, never `CLIPPED` or `LOWCONFIDENCE`.

String lengths for the clipping rule use the original extracted camera text and
OCR text. The different-reading check uses the same comparison normalization
as `SAME` / `DIFF` below. A small crop by itself does not count as detected
clipping. Filename or image/OCR errors remain `LOWCONFIDENCE` unless an exact
non-empty text match gives `SAME`; absent confidence
is not treated as below 60%, but a genuine OCR length mismatch can meet the rule.

Comparison ignores case, spaces and hyphens. It **does not** equate `O` with
`0`, `I` with `1`, or similar-looking characters. The original camera reading
and OCR text are retained in the output. The confidence score is the **minimum
character probability**, ignoring trailing model padding slots. It is not a
calibrated probability that the entire plate is correct, nor proof that a
high-confidence reading is ground truth. The threshold needs tuning against
your camera data.

Dates remain literal ISO-formatted text; no timezone or UTC conversion is
invented because the filename contains no timezone. All years are 2000–2099.
Plate readings are written as literal text (`RAW`), retaining leading zeroes
and preventing formula injection. The local CSV also guards formula prefixes.

## Cropping limitations

`fast-plate-ocr` expects **already cropped plate images**, not full vehicle or
road scenes. This app does not add an object detector.

The default crop guard looks for dark, character-sized connected components
touching any image edge and flags crops below 60 pixels wide or 20 pixels tall.
This is a **conservative heuristic**, not a trained cropping detector. It can
flag dirt, plate frames or shadows, and cannot reliably detect a character
completely missing outside the frame. Inspect questionable images, especially
high-confidence `DIFF` results. A manual crop flag counts as clipping for the
`CLIPPED` rule above; otherwise it gives `LOWCONFIDENCE`, except when an exact
non-empty text match gives `SAME`. Removing it does not override an automatic
crop flag for differing readings.

## Safety and recovery

- Source images are read only: never renamed, moved, overwritten or uploaded.
- Supported formats: JPG/JPEG, PNG, BMP, TIFF and WebP, case insensitive.
  For a multi-frame TIFF, OCR uses the first frame. Symlinks are skipped to
  avoid loops, out-of-folder traversal and duplicate references. Inaccessible
  subfolders stop discovery instead of silently reporting a complete scan.
- Inference and export run in background threads. **Stop** finishes the current
  image; it does not interrupt a model download. Partial reports can be exported
  after explicit confirmation.
- Local reports, explanations, per-character confidence and review flags are
  saved under `~/.local/share/plate-ocr-review/runs/<run-id>/` (or `$XDG_DATA_HOME`).
  `results.csv` has the exact five output columns; `report.json` has audit details.
- **Open saved report** loads a report without re-running OCR. After a crash,
  open `checkpoint.jsonl` to recover completed rows. Restart OCR for a new full
  scan; automatic continuation of inference is not implemented.
- A new spreadsheet is created by default. An optional existing Sheet URL
  creates a **new unique tab**, never replacing existing tabs.
- Sheet writes use 500-row batches, with read-back verification. Exporting again
  resumes the same run's destination without duplicate sheets/rows. Manual crop
  reviews can be re-exported to the same tab. External row edits cause a conflict
  rather than being overwritten.
- An ambiguous creation timeout is not retried blindly. Find the Sheet using
  the title in the error, paste its URL, and click **Recover export**. Existing
  sheet-tab creation can be resolved by retrying if the expected tab exists.
- The export includes **all** processed rows, irrespective of the UI filter.
  It does not change document sharing permissions or send email.

## Copy DIFF images

After an OCR run, or after opening a saved report, click **Copy DIFF images…**
and choose a destination folder. This copies **every result currently marked
DIFF**, regardless of the table filter. SAME, CLIPPED and LOWCONFIDENCE images
are excluded. Original files and report statuses remain unchanged.

The OCR result is added literally before the image extension:

```text
26-09-08_17-06-27_DOVER24_GSV81G_ABC.jpg
→ 26-09-08_17-06-27_DOVER24_GSV81G_ABC_<GSV81C>.jpg
```

- Copies are placed directly in the chosen folder, not in mirrored subfolders.
- Existing destination files are **never overwritten**. For conflicting names,
  a number is added before the OCR suffix, for example `photo_2_<ABC123>.jpg`.
- If an identical copy already exists, it is skipped. Copying again is safe.
- Copying runs in a background thread with progress. **Stop** finishes the current
  file and keeps completed copies. Missing/unreadable files are reported, and
  the remaining files are still attempted. Partial files from failed copies
  are removed when possible.
- A local JSON copy log is saved in the report's run directory; the destination
  folder contains only the copied images. No Google Sheet is changed.
- Prefer a destination outside the scanned source tree. If it is inside, the
  app asks for confirmation because future recursive scans will include copies.
- The requested `<` and `>` characters are valid on Linux filesystems, but not
  in Windows filenames or on some removable/network filesystems. Unsupported
  destinations report errors rather than silently changing the suffix.

## Install elsewhere / standalone Google sign-in

Python 3.11+ and a graphical desktop are required. CPU inference is the default.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
./run.sh
```

On Windows, activate `.venv` and run `python -m plate_ocr_review.app`.
On Linux, Qt may need system libraries such as `libxcb-cursor` and OpenGL;
headless test runs can use `QT_QPA_PLATFORM=offscreen`.

Without the local Workspace connector:

1. Enable the **Google Sheets API** in a Google Cloud project.
2. Configure its OAuth consent screen (add your account as a test user if needed).
3. Create an OAuth client with application type **Desktop app** and download
   its client JSON file. Keep it outside source control.
4. Select **Google OAuth** in the app and browse to that file.
5. Click export and complete the Google sign-in in your browser.

The standalone backend requests the narrow `drive.file` scope, which permits
creating and editing files authorized for this app. Arbitrary existing sheets
may not be accessible with that scope; use new sheets, previously created sheets,
or the Workspace connector rather than broadening permissions unnecessarily.
The standalone refresh token is stored locally with mode `0600` at
`~/.local/share/plate-ocr-review/oauth-token.json`. It is never put in a report.

## Tests

```bash
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/ruff check plate_ocr_review tests
```

Tests cover filename parsing, status/threshold rules, padding confidences,
recursive scanning, crop heuristics, unreadable images, report recovery,
literal five-column exports, interrupted writes, creation recovery, external
edit protection, and multi-batch exports. Google Sheet writes are mocked in
tests: running the suite does not create or modify remote documents.

Initial validation on this computer: 48 automated tests, native Qt desktop startup,
real CPU ONNX OCR on a synthetic plate (`GSV81G`, correctly recognised), and
a live read-only Google Sheets connector request. No source image folders
were processed and no Google Sheet was created during validation.

Filename configuration update: 69 automated tests passed, including custom
camera names, reordered fields, alternative separators, end-relative positions,
configuration persistence, older-report compatibility, and GUI preview validation.

CLIPPED status update: 89 automated tests passed. Tests verify the strict 60%
boundary, unequal string lengths, matching-reading exclusion, manual/automatic
clipping, small-image exclusion, UI filtering/counts and five-column Sheet export.
Existing reports and Google Sheets are not automatically reclassified or rewritten.

Exact-match priority update: 104 automated tests passed. Every non-empty exact
OCR/camera text match is classified as SAME, including low/missing confidence,
automatic/manual crop flags and processing warnings. Two empty readings are not
treated as a successful plate match.

DIFF image copying update: 120 automated tests passed, including literal OCR
suffixes, DIFF-only selection independent of the UI filter, filename conflicts,
repeat-copy detection, missing files, unsafe OCR text, symlinks, cancellation
and partial-copy cleanup. Tests only copy synthetic local files.
