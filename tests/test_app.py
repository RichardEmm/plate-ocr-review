import os
import subprocess
import sys
import time

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, QEvent

from plate_ocr_review import app as ui
from plate_ocr_review.core import FilenameConfig, Report, Result, Settings


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    win = ui.MainWindow()
    win.show()
    application.processEvents()
    yield win
    assert not win.busy
    if win.worker:
        assert win.worker.wait(1000)
    win.close()
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    application.processEvents()


def settle(application, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.005)
    assert predicate()
    application.processEvents()


def test_initial_controls(window):
    assert window.start.isEnabled()
    assert not window.export.isEnabled()
    assert not window.cancel.isEnabled()
    assert not window.copy_button.isEnabled()
    assert window.threshold.value() == 90
    assert window._filename_config() == FilenameConfig()
    assert "Camera: DOVER24" in window.filename_preview.text()
    assert "Plate: GSV81G" in window.filename_preview.text()


def test_filename_preview_and_preferences(window):
    config = FilenameConfig(camera_field=-2, plate_field=3)
    window._set_filename_config(config)
    window.filename_example.setText("26-09-08_17-06-27_00123_GateEast_suffix.jpg")
    assert "Camera: GateEast" in window.filename_preview.text()
    assert "Plate: 00123" in window.filename_preview.text()
    window._save_preferences()
    window._set_filename_config(FilenameConfig())
    window._load_preferences()
    assert window._filename_config() == config


def test_invalid_config_preview_and_run_rejected(window, tmp_path, monkeypatch):
    window.filename_fields["camera_field"].setValue(0)
    assert "Invalid filename configuration" in window.filename_preview.text()
    window.folder.setText(str(tmp_path))
    warnings = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window._start()
    assert warnings and "not 0" in warnings[0]
    assert not window.busy
    assert window.report is None


def test_scan_background_to_results(application, window, tmp_path, monkeypatch):
    folder = tmp_path / "images"
    folder.mkdir()
    path = folder / "26-09-08_17-06-27_DOVER24_ABC_X.jpg"
    path.touch()

    class FakeEngine:
        def __init__(self, settings):
            pass

        def process(self, path):
            row = Result(str(path), "2026-09-08 17:06:27", "DOVER24", "ABC", "ABC", [.99] * 3)
            row.classify(.9)
            return row

    monkeypatch.setattr(ui, "OcrEngine", FakeEngine)
    window.folder.setText(str(folder))
    window._start()
    assert window.busy
    assert not window.start.isEnabled()
    settle(application, lambda: not window.busy)
    assert window.report.complete
    assert window.model.rowCount() == 1
    assert window.export.isEnabled()
    assert (window.report.directory / "report.json").exists()


def test_filter_manual_crop_and_export(application, window, monkeypatch):
    rows = [Result("missing.jpg", camera_plate="ABC", ocr_plate="ABC", confidences=[.99] * 3),
            Result("missing2.jpg", camera_plate="ABC", ocr_plate="XYZ", confidences=[.99] * 3)]
    for row in rows:
        row.classify(.9)
    window.report = Report("images", Settings(), results=rows, complete=True, discovered=2)
    window.model.replace(rows)
    window._enable()
    window.filter.setCurrentText("SAME")
    assert window.proxy.rowCount() == 1
    window.table.selectRow(0)
    application.processEvents()
    assert window.manual_crop.isEnabled()
    window.manual_crop.setChecked(True)
    assert rows[0].status == "SAME"
    assert window.proxy.rowCount() == 1
    calls = []

    def fake_export(report, **kwargs):
        calls.append(report)
        return "https://docs.google.com/spreadsheets/d/1234567890123456789012345/edit"

    monkeypatch.setattr(ui, "run_export", fake_export)
    window._export()
    settle(application, lambda: not window.busy)
    assert len(calls[0].results) == 2  # Filter never limits Google Sheet output.
    assert window.open_sheet.isEnabled()


def test_clipped_filter_counter_and_color(window):
    result = Result("missing.jpg", camera_plate="ABC", ocr_plate="AB", confidences=[.99, .99],
                    crop_reasons=["Possible clipped character at right edge"])
    result.classify(.90)
    window.model.replace([result])
    window._stats()
    window.filter.setCurrentText("CLIPPED")
    assert window.proxy.rowCount() == 1
    assert window.stat_labels["CLIPPED"].text() == "1"
    assert window.model.data(window.model.index(0, 4), ui.Qt.ForegroundRole).name() == "#f5a36c"


def test_copy_diffs_button_copies_all_diff_rows_regardless_of_filter(application, window, tmp_path, monkeypatch):
    source = tmp_path / "images"
    source.mkdir()
    target = tmp_path / "copies"
    target.mkdir()
    rows = []
    for index, status in enumerate(["DIFF", "SAME", "DIFF", "CLIPPED", "LOWCONFIDENCE"]):
        path = source / f"{index}.jpg"
        path.write_bytes(f"image {index}".encode())
        rows.append(Result(str(path), ocr_plate="ABC123", status=status))
    window.report = Report(str(source), Settings(), results=rows, complete=True)
    window.model.replace(rows)
    window.filter.setCurrentText("SAME")
    window._enable()
    assert window.copy_button.isEnabled()
    monkeypatch.setattr(ui.QFileDialog, "getExistingDirectory", lambda *args: str(target))
    window._copy_diffs()
    assert window.busy
    assert not window.copy_button.isEnabled()
    settle(application, lambda: not window.busy)
    assert sorted(p.name for p in target.iterdir()) == ["0_<ABC123>.jpg", "2_<ABC123>.jpg"]
    assert "2 copied" in window.message.text()
    assert len(list(source.iterdir())) == 5


def test_cancel_preserves_partial_report(application, window, tmp_path, monkeypatch):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(50):
        (folder / f"{i}.jpg").touch()

    class SlowEngine:
        def __init__(self, settings):
            pass

        def process(self, path):
            time.sleep(.01)
            return Result(str(path), error="Bad filename")

    monkeypatch.setattr(ui, "OcrEngine", SlowEngine)
    window.folder.setText(str(folder))
    window._start()
    settle(application, lambda: len(window.report.results) >= 1)
    window._cancel()
    settle(application, lambda: not window.busy)
    assert 0 < len(window.report.results) < 50
    assert not window.report.complete
    assert (window.report.directory / "results.csv").exists()


@pytest.mark.parametrize("mode", ["idle", "after-worker"])
def test_real_entrypoint_closes_cleanly(tmp_path, mode):
    script = '''
import os
import sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from plate_ocr_review import app as ui
from plate_ocr_review.core import Result

class FakeEngine:
    def __init__(self, settings):
        pass
    def process(self, path):
        return Result(str(path), error="Synthetic shutdown test")

ui.OcrEngine = FakeEngine
original = ui.MainWindow.__init__
def initialize(self):
    original(self)
    QTimer.singleShot(3000, lambda: QApplication.instance().exit(9))
    if sys.argv[-1] == "idle":
        QTimer.singleShot(50, self.close)
    else:
        folder = Path(os.environ["XDG_DATA_HOME"]) / "images"
        folder.mkdir(parents=True)
        (folder / "test.jpg").touch()
        self.folder.setText(str(folder))
        def scan():
            self._start()
            self.worker.finished.connect(self.close)
        QTimer.singleShot(0, scan)
ui.MainWindow.__init__ = initialize
ui.main()
'''
    result = subprocess.run([sys.executable, "-c", script, mode], capture_output=True,
        text=True, timeout=10, env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
                                   "XDG_DATA_HOME": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert "shared QObject was deleted directly" not in result.stderr
