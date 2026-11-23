"""Qt desktop interface; all inference and network work runs off the UI thread."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import (
    QAbstractTableModel, QCoreApplication, QEvent, QModelIndex, QSortFilterProxyModel,
    Qt, QThread, QUrl, Signal,
)
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableView, QVBoxLayout, QWidget,
)

from .core import (
    HEADERS, MODELS, FilenameConfig, OcrEngine, Report, Settings, atomic_json,
    data_directory, discover_images, parse_filename,
)
from .sheets import run_export
from .copying import copy_diff_images

STYLE = """
QMainWindow, QWidget { background: #0d1622; color: #dce7f3; font-family: 'DejaVu Sans'; font-size: 13px; }
QFrame#card { background: #152232; border: 1px solid #26394d; border-radius: 10px; }
QFrame#card QLabel, QFrame#card QCheckBox { background: transparent; }
QLabel#eyebrow { color: #50d9bd; font-size: 11px; font-weight: bold; }
QLabel#title { font-size: 29px; font-weight: bold; }
QLabel#muted { color: #91a6bc; }
QLabel#stat { font-size: 23px; font-weight: bold; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: #0f1b2a; border: 1px solid #32485f;
  border-radius: 6px; padding: 8px; selection-background-color: #287a71; }
QPushButton { background: #22374b; border: 1px solid #38516b; border-radius: 6px;
  padding: 9px 15px; font-weight: bold; }
QPushButton:hover { background: #2c455c; }
QPushButton#primary { background: #50d9bd; color: #08251e; border: none; }
QPushButton#primary:hover { background: #74e6cf; }
QPushButton:disabled { background: #1a293a; color: #526b84; border-color: #26394d; }
QPushButton#primary:disabled { background: #1a293a; color: #526b84; border: 1px solid #26394d; }
QTableView { background: #101c2b; alternate-background-color: #152334; border: 1px solid #26394d;
  gridline-color: #26394d; selection-background-color: #234d59; border-radius: 6px; }
QHeaderView::section { background: #1e3044; color: #aebfd0; padding: 10px 7px;
  border: none; border-right: 1px solid #2b4055; font-weight: bold; }
QProgressBar { border: none; border-radius: 4px; background: #22364b; height: 7px; }
QProgressBar::chunk { background: #50d9bd; border-radius: 4px; }
QSplitter::handle { background: #26394d; }
QCheckBox { spacing: 7px; }
QToolTip { background: #22374b; color: #e2eaf2; border: 1px solid #50d9bd; }
"""


class ResultsModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.results)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 6

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return ["Date/Time", "Camera", "Camera reading", "OCR reading", "Status", "Min. confidence"][section]
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal:
            return (HEADERS + ["Minimum per-character confidence"])[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        result = self.results[index.row()]
        if role == Qt.DisplayRole:
            return (result.sheet_row() + [f"{result.confidence:.1%}"])[index.column()]
        if role == Qt.ForegroundRole and index.column() == 4:
            return QColor({"SAME": "#50d9bd", "DIFF": "#ff9292", "CLIPPED": "#f5a36c",
                           "LOWCONFIDENCE": "#f1c878"}[result.status])
        if role == Qt.ToolTipRole:
            return result.path + "\n" + result.explanation
        return None

    def replace(self, results):
        self.beginResetModel()
        self.results = list(results)
        self.endResetModel()

    def append(self, results):
        if results:
            start = len(self.results)
            self.beginInsertRows(QModelIndex(), start, start + len(results) - 1)
            self.results.extend(results)
            self.endInsertRows()


class ScanWorker(QThread):
    message = Signal(str)
    batch = Signal(list)
    progress = Signal(int, int)
    completed = Signal(bool, str)

    def __init__(self, report: Report, parent=None):
        super().__init__(parent)
        self.report = report
        self.cancel = threading.Event()

    def run(self):
        pending = []
        error = ""
        try:
            self.message.emit("Discovering images in the folder and all subfolders…")
            files = discover_images(Path(self.report.folder), self.cancel.is_set)
            self.report.discovered = len(files)
            atomic_json(self.report.directory / "manifest.json", asdict(self.report))
            if self.cancel.is_set():
                return
            if not files:
                raise ValueError("No supported images found in this folder")
            self.message.emit("Loading OCR model (first use downloads the model)…")
            engine = OcrEngine(self.report.settings)
            started = time.monotonic()
            last_emit = started
            for index, path in enumerate(files):
                if self.cancel.is_set():
                    break
                result = engine.process(path)
                self.report.results.append(result)
                self.report.checkpoint(result)
                pending.append(result)
                now = time.monotonic()
                if len(pending) >= 25 or now - last_emit >= .25:
                    self.batch.emit(pending)
                    pending = []
                    self.progress.emit(index + 1, len(files))
                    elapsed = now - started
                    eta = elapsed / (index + 1) * (len(files) - index - 1)
                    self.message.emit(f"{index+1:,} / {len(files):,} images  ·  About {eta/60:.1f} min remaining")
                    last_emit = now
            self.report.complete = len(self.report.results) == len(files) and not self.cancel.is_set()
        except Exception as exc:
            error = str(exc)
        finally:
            if pending:
                self.batch.emit(pending)
            try:
                self.report.save()
            except Exception as exc:
                error = f"{error}\nCould not save local report: {exc}".strip()
            self.completed.emit(self.report.complete, error)


class ExportWorker(QThread):
    message = Signal(str)
    completed = Signal(str, str)

    def __init__(self, report, options, recover=False, parent=None):
        super().__init__(parent)
        self.report = report
        self.options = options
        self.recover = recover

    def run(self):
        try:
            url = run_export(self.report, **self.options, progress=self.message.emit, recover=self.recover)
            self.completed.emit(url, "")
        except BaseException as exc:
            # MCP can raise an ExceptionGroup; retain nested messages for diagnosis.
            def describe(error):
                if isinstance(error, BaseExceptionGroup):
                    return "\n".join(describe(child) for child in error.exceptions)
                return str(error)
            self.completed.emit("", describe(exc))


class CopyWorker(QThread):
    progress = Signal(int, int)
    completed = Signal(object, str)

    def __init__(self, report, destination, parent=None):
        super().__init__(parent)
        self.report = report
        self.destination = destination
        self.cancel = threading.Event()

    def run(self):
        try:
            summary = copy_diff_images(self.report, self.destination,
                                       self.progress.emit, self.cancel.is_set)
            self.completed.emit(summary, "")
        except Exception as exc:
            self.completed.emit(None, str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plate OCR Review")
        self.resize(1420, 950)
        self.setMinimumSize(1050, 650)
        self.report = None
        self.worker = None
        self.preview_pixmap = None
        self.busy = False
        self.sheet_url = ""
        self.copy_destination = ""
        self.model = ResultsModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(4)
        self._build()
        self._load_preferences()
        self._filename_preview()
        self._enable()

    def _build(self):
        central = QWidget()
        # Keep controls at their natural size on smaller/scaled desktops. A
        # scrollable workspace is preferable to clipping buttons and review flags.
        central.setMinimumSize(1100, 1120)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(26, 22, 26, 20)
        layout.setSpacing(16)
        eyebrow = QLabel("LOCAL OCR   /   GOOGLE SHEETS")
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)
        header = QHBoxLayout()
        title = QLabel("Plate OCR Review")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.load_button = self._button("Open saved report", self._load_report)
        header.addWidget(self.load_button)
        layout.addLayout(header)
        layout.addWidget(self._muted("Compare camera readings with fast-plate-ocr. Your source images are never changed."))

        card = self._card()
        box = QVBoxLayout(card)
        box.setContentsMargins(18, 14, 18, 14)
        box.addWidget(QLabel("01   IMAGE SOURCE"))
        source = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Choose a folder — every subfolder is included")
        source.addWidget(self.folder, 1)
        self.browse = self._button("Choose folder…", self._choose_folder)
        source.addWidget(self.browse)
        self.start = self._button("Run OCR", self._start, primary=True)
        source.addWidget(self.start)
        self.cancel = self._button("Stop", self._cancel)
        source.addWidget(self.cancel)
        box.addLayout(source)
        options = QHBoxLayout()
        options.addWidget(QLabel("Model"))
        self.model_choice = QComboBox()
        self.model_choice.addItems(MODELS)
        self.model_choice.setMinimumWidth(245)
        options.addWidget(self.model_choice)
        options.addSpacing(14)
        options.addWidget(QLabel("High confidence ≥"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(1, 100)
        self.threshold.setValue(90)
        self.threshold.setSuffix(" %")
        self.threshold.setDecimals(1)
        options.addWidget(self.threshold)
        self.crop_check = QCheckBox("Flag likely clipped characters")
        self.crop_check.setChecked(True)
        self.crop_check.setToolTip("Conservative edge heuristic; not a guarantee that cropping will be detected")
        options.addWidget(self.crop_check)
        options.addStretch()
        box.addLayout(options)
        box.addWidget(self._muted("Exact OCR/camera text matches always give SAME. Otherwise confidence is the weakest recognised character."))
        box.addWidget(QLabel("FILENAME CONFIGURATION"))
        filename_options = QHBoxLayout()
        filename_options.addWidget(QLabel("Separator"))
        self.filename_separator = QLineEdit("_")
        self.filename_separator.setMaxLength(16)
        self.filename_separator.setMaximumWidth(80)
        filename_options.addWidget(self.filename_separator)
        self.filename_fields = {}
        for key, label, default in [("camera_field", "Camera field", 3),
                                    ("plate_field", "Plate field", 4),
                                    ("date_field", "Date field", 1), ("time_field", "Time field", 2)]:
            filename_options.addWidget(QLabel(label))
            control = QSpinBox()
            control.setRange(-100, 100)
            control.setValue(default)
            control.setMinimumWidth(80)
            control.setToolTip("1 = first field; -1 = last field. Zero is invalid. Extension is excluded.")
            self.filename_fields[key] = control
            filename_options.addWidget(control)
        filename_options.addStretch()
        box.addLayout(filename_options)
        box.addWidget(self._muted("Fields count from 1. Negative positions count from the end (-1 = last). Camera names are unrestricted."))
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Test filename"))
        self.filename_example = QLineEdit("26-09-08_17-06-27_DOVER24_GSV81G_ABC.jpg")
        preview_row.addWidget(self.filename_example, 1)
        self.filename_example_browse = self._button("Choose example…", self._choose_filename_example)
        preview_row.addWidget(self.filename_example_browse)
        box.addLayout(preview_row)
        self.filename_preview = self._muted("")
        self.filename_preview.setTextFormat(Qt.PlainText)
        self.filename_preview.setWordWrap(True)
        box.addWidget(self.filename_preview)
        self.filename_separator.textChanged.connect(self._filename_preview)
        self.filename_example.textChanged.connect(self._filename_preview)
        for control in self.filename_fields.values():
            control.valueChanged.connect(self._filename_preview)
        layout.addWidget(card)

        stats = QHBoxLayout()
        self.stat_labels = {}
        for key, text, color in [("total", "PROCESSED", "#dce7f3"), ("SAME", "SAME", "#50d9bd"),
                                 ("DIFF", "DIFF", "#ff9292"), ("CLIPPED", "CLIPPED", "#f5a36c"),
                                 ("LOWCONFIDENCE", "LOW CONFIDENCE", "#f1c878")]:
            stat_card = self._card()
            stat_box = QVBoxLayout(stat_card)
            stat_box.setContentsMargins(16, 12, 16, 12)
            stat_box.addWidget(self._muted(text))
            number = QLabel("0")
            number.setObjectName("stat")
            number.setStyleSheet(f"color: {color}; background: transparent;")
            self.stat_labels[key] = number
            stat_box.addWidget(number)
            stats.addWidget(stat_card)
        layout.addLayout(stats)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("RESULTS"))
        self.filter = QComboBox()
        self.filter.addItems(["All results", "DIFF", "SAME", "CLIPPED", "LOWCONFIDENCE"])
        self.filter.currentTextChanged.connect(self._filter)
        toolbar.addWidget(self.filter)
        toolbar.addStretch()
        self.copy_button = self._button("Copy DIFF images…", self._copy_diffs)
        self.copy_button.setToolTip("Copies all DIFF images, regardless of the table filter; originals remain unchanged")
        toolbar.addWidget(self.copy_button)
        self.csv_button = self._button("Save CSV…", self._save_csv)
        toolbar.addWidget(self.csv_button)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for index, width in enumerate([175, 90, 145, 145, 150, 125]):
            self.table.setColumnWidth(index, width)
        self.table.selectionModel().selectionChanged.connect(self._selection)
        splitter.addWidget(self.table)
        review = self._card()
        review.setMinimumWidth(270)
        review.setMinimumHeight(255)
        review_box = QVBoxLayout(review)
        review_box.setContentsMargins(16, 14, 16, 14)
        review_box.addWidget(QLabel("IMAGE REVIEW"))
        self.image = QLabel("Select a result to preview its image")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(230, 100)
        self.image.setMaximumHeight(170)
        self.image.setWordWrap(True)
        self.image.setStyleSheet("background: #0b1420; border-radius: 6px; color: #91a6bc;")
        review_box.addWidget(self.image)
        self.filename = self._muted("")
        self.filename.setWordWrap(True)
        self.filename.setTextFormat(Qt.PlainText)
        review_box.addWidget(self.filename)
        self.reason = QLabel("Exact OCR/camera text matches always give SAME. Clipping rules apply to different readings.")
        self.reason.setWordWrap(True)
        self.reason.setTextFormat(Qt.PlainText)
        review_box.addWidget(self.reason)
        self.manual_crop = QCheckBox("Clipped / overly cropped (manual flag)")
        self.manual_crop.toggled.connect(self._manual_crop)
        review_box.addWidget(self.manual_crop)
        review_box.addStretch()
        splitter.addWidget(review)
        splitter.setSizes([960, 300])
        layout.addWidget(splitter, 1)

        sheet_card = self._card()
        sheet_box = QVBoxLayout(sheet_card)
        sheet_box.setContentsMargins(18, 14, 18, 14)
        sheet_box.addWidget(QLabel("02   GOOGLE SHEETS EXPORT"))
        export_row = QHBoxLayout()
        self.sheet_title = QLineEdit("ANPR OCR comparison")
        self.sheet_title.setPlaceholderText("New spreadsheet title")
        export_row.addWidget(self.sheet_title, 1)
        self.destination = QLineEdit()
        self.destination.setPlaceholderText("Optional existing Sheet URL — creates a new tab")
        export_row.addWidget(self.destination, 2)
        self.export = self._button("Export to Google Sheets", self._export, primary=True)
        export_row.addWidget(self.export)
        self.open_sheet = self._button("Open Sheet ↗", self._open_sheet)
        export_row.addWidget(self.open_sheet)
        sheet_box.addLayout(export_row)
        auth = QHBoxLayout()
        self.backend = QComboBox()
        self.backend.addItems(["Workspace connector", "Google OAuth"])
        self.backend.setMinimumWidth(200)
        self.backend.currentTextChanged.connect(self._auth_changed)
        auth.addWidget(self.backend)
        self.auth_path = QLineEdit(str(Path.home() / ".local/bin/google-workspace-mcp"))
        auth.addWidget(self.auth_path, 1)
        self.auth_browse = self._button("Browse…", self._choose_auth)
        auth.addWidget(self.auth_browse)
        self.recover = self._button("Recover export", lambda: self._export(recover=True))
        self.recover.setToolTip("Recover a timed-out sheet creation using the original sheet URL")
        auth.addWidget(self.recover)
        sheet_box.addLayout(auth)
        sheet_box.addWidget(self._muted("Exports all results, regardless of the table filter. Five columns only. Uploads text, never images."))
        layout.addWidget(sheet_card)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.message = self._muted("Ready. Choose an image folder to begin.")
        self.message.setTextFormat(Qt.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

    @staticmethod
    def _card():
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    @staticmethod
    def _muted(text):
        label = QLabel(text)
        label.setObjectName("muted")
        return label

    @staticmethod
    def _button(text, action, primary=False):
        button = QPushButton(text)
        button.clicked.connect(action)
        if primary:
            button.setObjectName("primary")
        return button

    def _choose_folder(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose ANPR image folder", self.folder.text())
        if chosen:
            self.folder.setText(chosen)

    def _choose_auth(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose connector launcher or OAuth client JSON")
        if path:
            self.auth_path.setText(path)

    def _choose_filename_example(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose image to test filename configuration", self.folder.text())
        if path:
            self.filename_example.setText(Path(path).name)

    def _filename_config(self):
        return FilenameConfig(separator=self.filename_separator.text(), **{
            key: widget.value() for key, widget in self.filename_fields.items()})

    def _set_filename_config(self, config):
        self.filename_separator.setText(config.separator)
        for key, control in self.filename_fields.items():
            control.setValue(getattr(config, key))
        self._filename_preview()

    def _filename_preview(self, *_):
        try:
            config = self._filename_config()
            example = Path(self.filename_example.text())
            meta = parse_filename(example, config)
            fields = "  ·  ".join(f"{i}: {part}" for i, part in enumerate(
                example.stem.split(config.separator), 1))
            summary = f"Camera: {meta.camera or '—'}  ·  Plate: {meta.camera_plate or '—'}  ·  Date/Time: {meta.timestamp or '—'}"
            self.filename_preview.setText(fields + "\n" + summary +
                                           ("\nWarning: " + meta.error if meta.error else ""))
        except ValueError as exc:
            self.filename_preview.setText("Invalid filename configuration: " + str(exc))

    def _auth_changed(self, value):
        self.auth_path.setText(str(Path.home() / ".local/bin/google-workspace-mcp")
                               if value == "Workspace connector" else "")
        self.auth_path.setPlaceholderText("Google OAuth Desktop app client JSON file")

    def _enable(self):
        has_results = self.report is not None and bool(self.report.results)
        for widget in [self.folder, self.browse, self.start, self.threshold, self.crop_check,
                       self.model_choice, self.load_button, self.sheet_title, self.destination,
                       self.backend, self.auth_path, self.auth_browse, self.filename_separator,
                       self.filename_example, self.filename_example_browse, *self.filename_fields.values()]:
            widget.setEnabled(not self.busy)
        for widget in [self.csv_button, self.export, self.recover]:
            widget.setEnabled(has_results and not self.busy)
        self.copy_button.setEnabled(has_results and not self.busy and any(
            result.status == "DIFF" for result in self.report.results))
        self.manual_crop.setEnabled(self._selected() is not None and not self.busy)
        self.cancel.setEnabled(self.busy and isinstance(self.worker, (ScanWorker, CopyWorker)))
        self.open_sheet.setEnabled(bool(self.sheet_url))

    def _start(self):
        folder = Path(self.folder.text()).expanduser()
        if not self.folder.text().strip() or not folder.is_dir():
            QMessageBox.warning(self, "Choose a folder", "Select an existing ANPR image folder first.")
            return
        if self.report and self.report.results:
            if QMessageBox.question(self, "Start new run?", "The current report is saved locally. Start a new OCR run?") != QMessageBox.Yes:
                return
        try:
            settings = Settings(self.model_choice.currentText(), self.threshold.value() / 100,
                                self.crop_check.isChecked(), self._filename_config())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid filename configuration", str(exc))
            return
        self._save_preferences()
        self.report = Report(str(folder.resolve()), settings)
        self.model.replace([])
        self.sheet_url = ""
        self._clear_preview()
        self._stats()
        self.busy = True
        self.progress.setRange(0, 0)
        if self.worker:
            self.worker.deleteLater()
        self.worker = ScanWorker(self.report, self)
        self.worker.message.connect(self.message.setText)
        self.worker.batch.connect(self._batch)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._scan_done)
        self.worker.finished.connect(self._worker_finished)
        self._enable()
        self.worker.start()

    def _worker_finished(self):
        # Do not permit a second job until the preceding QThread has actually exited.
        self.busy = False
        self._enable()

    def _batch(self, results):
        self.model.append(results)
        self._stats()

    def _stats(self):
        counts = Counter(r.status for r in self.model.results)
        self.stat_labels["total"].setText(f"{len(self.model.results):,}")
        for status in ["SAME", "DIFF", "CLIPPED", "LOWCONFIDENCE"]:
            self.stat_labels[status].setText(f"{counts[status]:,}")

    def _progress(self, count, total):
        self.progress.setRange(0, total)
        self.progress.setValue(count)

    def _scan_done(self, complete, error):
        self.model.replace(self.report.results)
        self._stats()
        self.progress.setRange(0, max(1, self.report.discovered))
        self.progress.setValue(len(self.report.results))
        state = "Completed" if complete else "Stopped — partial report"
        self.message.setText(f"{state}: {len(self.report.results):,} images. Saved in {self.report.directory}")
        if error:
            QMessageBox.warning(self, "OCR run stopped", error)

    def _cancel(self):
        if isinstance(self.worker, (ScanWorker, CopyWorker)):
            self.worker.cancel.set()
            self.cancel.setEnabled(False)
            self.message.setText(
                "Stopping after the current file; completed copies will be kept…"
                if isinstance(self.worker, CopyWorker) else
                "Stopping after the current image/model download; completed results will be saved…")

    def _filter(self, value):
        self.proxy.setFilterFixedString("" if value == "All results" else value)

    def _selected(self):
        indexes = self.table.selectionModel().selectedRows()
        if indexes:
            source = self.proxy.mapToSource(indexes[0])
            if source.isValid():
                return self.model.results[source.row()]
        return None

    def _clear_preview(self):
        self.image.clear()
        self.image.setText("Select a result to preview its image")
        self.preview_pixmap = None
        self.filename.setText("")
        self.reason.setText("Exact OCR/camera text matches always give SAME. Clipping rules apply to different readings.")
        self.manual_crop.blockSignals(True)
        self.manual_crop.setChecked(False)
        self.manual_crop.blockSignals(False)

    def _selection(self, *_):
        result = self._selected()
        if result is None:
            self._clear_preview()
            self._enable()
            return
        self.filename.setText(result.path)
        self.reason.setText(result.explanation)
        self.manual_crop.blockSignals(True)
        self.manual_crop.setChecked(result.manual_crop)
        self.manual_crop.blockSignals(False)
        try:
            with Image.open(result.path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((1000, 500))
            qt_image = QImage(image.tobytes(), image.width, image.height,
                              image.width * 3, QImage.Format_RGB888).copy()
            self.preview_pixmap = QPixmap.fromImage(qt_image)
            self._scale_preview()
        except Exception as exc:
            self.preview_pixmap = None
            self.image.setText(f"Cannot preview image: {exc}")
        self._enable()

    def _scale_preview(self):
        if self.preview_pixmap:
            self.image.setPixmap(self.preview_pixmap.scaled(
                self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "image"):
            self._scale_preview()

    def _manual_crop(self, checked):
        result = self._selected()
        if result is None or self.busy:
            return
        original = result.manual_crop
        result.manual_crop = checked
        result.classify(self.report.settings.threshold)
        try:
            self.report.save()
        except Exception as exc:
            result.manual_crop = original
            result.classify(self.report.settings.threshold)
            QMessageBox.warning(self, "Could not save review", str(exc))
        self.model.dataChanged.emit(self.model.index(0, 0), self.model.index(self.model.rowCount()-1, 5))
        self._stats()
        self._selection()
        self.message.setText("Review saved locally. Export again to update this run's Google Sheet.")

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save results CSV", "plate-ocr-results.csv", "CSV (*.csv)")
        if path:
            try:
                self.report.save_csv(Path(path))
                self.message.setText(f"CSV saved: {path}")
            except Exception as exc:
                QMessageBox.warning(self, "CSV export failed", str(exc))

    def _copy_diffs(self):
        if self.busy or not self.report or not any(r.status == "DIFF" for r in self.report.results):
            return
        chosen = QFileDialog.getExistingDirectory(self, "Choose destination folder for DIFF images",
                                                   self.copy_destination)
        if not chosen:
            return
        destination = Path(chosen).resolve()
        if destination.is_relative_to(Path(self.report.folder).expanduser().resolve()):
            if QMessageBox.question(self, "Destination is inside the image source",
                "Future scans of this folder tree will include the copies. Continue?") != QMessageBox.Yes:
                return
        if not self.report.complete and QMessageBox.question(self, "Copy from partial report?",
            "This scan is incomplete. Copy all DIFF images processed so far?") != QMessageBox.Yes:
            return
        self.copy_destination = str(destination)
        self._save_preferences()
        if self.worker:
            self.worker.deleteLater()
        self.worker = CopyWorker(self.report, destination, self)
        self.worker.progress.connect(self._copy_progress)
        self.worker.completed.connect(self._copy_done)
        self.worker.finished.connect(self._worker_finished)
        count = sum(r.status == "DIFF" for r in self.report.results)
        self.busy = True
        self.progress.setRange(0, count)
        self.progress.setValue(0)
        self.message.setText(f"Copying {count:,} DIFF images to {destination}…")
        self._enable()
        self.worker.start()

    def _copy_progress(self, count, total):
        self._progress(count, total)
        self.message.setText(f"Copying DIFF images: {count:,} / {total:,} files checked")

    def _copy_done(self, summary, error):
        if error:
            self.message.setText("DIFF copy incomplete. Originals and existing destination files were not overwritten.")
            QMessageBox.warning(self, "Could not complete DIFF copy", error)
            return
        state = "Stopped" if summary.cancelled else "Finished"
        self.message.setText(
            f"{state}: {summary.copied:,} copied, {summary.already_present:,} already present, "
            f"{len(summary.errors):,} failed. Destination: {summary.destination}. Log: {summary.receipt}")
        if summary.errors:
            details = "\n".join(summary.errors[:10])
            if len(summary.errors) > 10:
                details += f"\n…and {len(summary.errors)-10} more; see the copy log."
            QMessageBox.warning(self, "Some DIFF files could not be copied", details)

    def _load_report(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open report or recover checkpoint",
            str(data_directory() / "runs"), "Saved report (report.json checkpoint.jsonl)")
        if not path:
            return
        try:
            report = Report.load(Path(path))
            self.report = report
            self.folder.setText(report.folder)
            self.model_choice.setCurrentText(report.settings.model)
            self.threshold.setValue(report.settings.threshold * 100)
            self.crop_check.setChecked(report.settings.crop_check)
            self._set_filename_config(report.settings.filename)
            self.model.replace(report.results)
            receipt = report.directory / "sheet-receipt.json"
            self.sheet_url = json.loads(receipt.read_text()).get("url", "") if receipt.exists() else ""
            self._clear_preview()
            self._stats()
            self._enable()
            self.message.setText(f"Opened {len(report.results):,} results. {'Complete' if report.complete else 'Partial'} report.")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open report", str(exc))

    def _export(self, _checked=False, recover=False):
        if not self.report or not self.report.results or self.busy:
            return
        if not self.report.complete and QMessageBox.question(self, "Export partial report?",
            f"Only {len(self.report.results):,} of {self.report.discovered:,} discovered images were processed. Export these results?") != QMessageBox.Yes:
            return
        self._save_preferences()
        self.busy = True
        options = {"backend": self.backend.currentText(), "connector": self.auth_path.text(),
                   "client_file": self.auth_path.text(), "title": self.sheet_title.text(),
                   "target": self.destination.text()}
        if self.worker:
            self.worker.deleteLater()
        self.worker = ExportWorker(self.report, options, recover, self)
        self.worker.message.connect(self.message.setText)
        self.worker.completed.connect(self._export_done)
        self.worker.finished.connect(self._worker_finished)
        self.progress.setRange(0, 0)
        self.message.setText("Connecting to Google Sheets. If sign-in is needed, complete it in your browser…")
        self._enable()
        self.worker.start()

    def _export_done(self, url, error):
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if url else 0)
        if error:
            self.message.setText("Export incomplete. Local OCR results are safe; retry resumes the same destination.")
            QMessageBox.warning(self, "Google Sheets export incomplete", error +
                "\n\nFor connector sign-in, run ~/.local/bin/google-workspace-login in a local terminal.")
        else:
            self.sheet_url = url
            self.message.setText(f"Google Sheet verified: {len(self.report.results):,} image rows, five columns. {url}")

    def _open_sheet(self):
        if self.sheet_url:
            QDesktopServices.openUrl(QUrl(self.sheet_url))

    def _save_preferences(self):
        try:
            atomic_json(data_directory() / "preferences.json", {
                "folder": self.folder.text(), "model": self.model_choice.currentText(),
                "threshold": self.threshold.value(), "crop": self.crop_check.isChecked(),
                "backend": self.backend.currentText(), "auth_path": self.auth_path.text(),
                "title": self.sheet_title.text(),
                "filename": {"separator": self.filename_separator.text(), **{
                    key: control.value() for key, control in self.filename_fields.items()}},
                "filename_example": self.filename_example.text(),
                "copy_destination": self.copy_destination,
            })
        except OSError:
            pass

    def _load_preferences(self):
        try:
            values = json.loads((data_directory() / "preferences.json").read_text())
            self.folder.setText(values.get("folder", ""))
            self.model_choice.setCurrentText(values.get("model", MODELS[0]))
            self.threshold.setValue(values.get("threshold", 90))
            self.crop_check.setChecked(values.get("crop", True))
            self.backend.setCurrentText(values.get("backend", "Workspace connector"))
            self.auth_path.setText(values.get("auth_path", self.auth_path.text()))
            self.sheet_title.setText(values.get("title", "ANPR OCR comparison"))
            self._set_filename_config(FilenameConfig(**values.get("filename", {})))
            self.filename_example.setText(values.get("filename_example", self.filename_example.text()))
            self.copy_destination = values.get("copy_destination", "")
        except (OSError, ValueError):
            pass

    def closeEvent(self, event):
        if self.busy:
            event.ignore()
            if isinstance(self.worker, ScanWorker):
                if QMessageBox.question(self, "Stop OCR?", "Stop after the current image and save partial results?") == QMessageBox.Yes:
                    self._cancel()
            elif isinstance(self.worker, CopyWorker):
                if QMessageBox.question(self, "Stop copying?", "Stop after the current file and keep completed copies?") == QMessageBox.Yes:
                    self._cancel()
            else:
                QMessageBox.information(self, "Export in progress", "Please wait for the current Google Sheets operation before closing.")
            return
        self._save_preferences()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    exit_code = app.exec()
    # Dispose of the Qt object tree while QApplication and Python wrappers are
    # still alive. Leaving this to interpreter teardown can crash PySide6.
    if window.worker:
        if isinstance(window.worker, (ScanWorker, CopyWorker)):
            window.worker.cancel.set()
        window.worker.wait()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
