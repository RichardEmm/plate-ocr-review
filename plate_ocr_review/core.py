"""Filename parsing, conservative classification, image OCR, and durable reports."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

HEADERS = ["Date/Time", "Camera", "Camera interpretation", "fast-plate-ocr interpretation", "Status"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODELS = ["cct-s-v2-global-model", "cct-xs-v2-global-model"]


def data_directory() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "plate-ocr-review"


def normalize_plate(value: str) -> str:
    """Ignore case, spaces, and separators, but never equate O/0 or I/1."""
    return re.sub(r"[\s_-]", "", value).upper()


@dataclass
class FilenameConfig:
    """Literal separator and 1-based fields; negative positions count from the end."""

    separator: str = "_"
    camera_field: int = 3
    plate_field: int = 4
    date_field: int = 1
    time_field: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.separator, str) or not self.separator or len(self.separator) > 16:
            raise ValueError("Filename separator must contain 1–16 characters")
        positions = [self.camera_field, self.plate_field, self.date_field, self.time_field]
        if any(type(position) is not int or position == 0 or abs(position) > 100
               for position in positions):
            raise ValueError("Filename fields must be integers from -100 to -1 or 1 to 100 (not 0)")
        if len(set(positions)) != len(positions):
            raise ValueError("Date, time, camera and plate must use different filename fields")


@dataclass
class Metadata:
    timestamp: str = ""
    camera: str = ""
    camera_plate: str = ""
    error: str = ""


def parse_filename(path: Path, config: FilenameConfig | None = None) -> Metadata:
    config = config or FilenameConfig()
    parts = path.stem.split(config.separator)
    errors = []
    selected = {}
    resolved = {}
    for name, position in [("camera", config.camera_field), ("plate", config.plate_field),
                           ("date", config.date_field), ("time", config.time_field)]:
        index = position - 1 if position > 0 else len(parts) + position
        if not 0 <= index < len(parts):
            errors.append(f"Missing {name} field {position}; filename has {len(parts)} fields")
            selected[name] = ""
        else:
            selected[name] = parts[index].strip()
            if not selected[name]:
                errors.append(f"Empty {name} field {position}")
            if index in resolved:
                errors.append(f"{name.capitalize()} and {resolved[index]} select the same filename field")
            resolved[index] = name
    result = Metadata(camera=selected["camera"], camera_plate=selected["plate"])
    if selected["plate"] and not normalize_plate(selected["plate"]):
        errors.append("Camera plate field contains no plate characters")
    try:
        if not all(re.fullmatch(r"\d{2}-\d{2}-\d{2}", selected[name]) for name in ["date", "time"]):
            raise ValueError("Date/time format")
        yy, month, day = map(int, selected["date"].split("-"))
        hour, minute, second = map(int, selected["time"].split("-"))
        result.timestamp = datetime(2000 + yy, month, day, hour, minute, second).isoformat(" ")
    except ValueError:
        errors.append("Invalid capture date/time: expected YY-MM-DD and HH-MM-SS in configured fields")
    result.error = "; ".join(errors)
    return result


@dataclass
class Settings:
    model: str = MODELS[0]
    threshold: float = 0.90
    crop_check: bool = True
    filename: FilenameConfig = field(default_factory=FilenameConfig)

    def __post_init__(self) -> None:
        if isinstance(self.filename, dict):
            self.filename = FilenameConfig(**self.filename)
        if not isinstance(self.filename, FilenameConfig):
            raise ValueError("Invalid filename configuration")
        if not 0 < self.threshold <= 1:
            raise ValueError("Confidence threshold must be greater than 0 and at most 1")
        if self.model not in MODELS:
            raise ValueError("Unsupported model")


@dataclass
class Result:
    path: str
    timestamp: str = ""
    camera: str = ""
    camera_plate: str = ""
    ocr_plate: str = ""
    confidences: list[float] = field(default_factory=list)
    confidence: float = 0.0
    crop_reasons: list[str] = field(default_factory=list)
    error: str = ""
    manual_crop: bool = False
    status: str = "LOWCONFIDENCE"

    @property
    def detected_clipping(self) -> bool:
        return self.manual_crop or any(
            reason.startswith("Possible clipped character at ") for reason in self.crop_reasons
        )

    def classify(self, threshold: float) -> None:
        valid_probs = (
            bool(self.confidences)
            and len(self.confidences) == len(self.ocr_plate)
            and all(math.isfinite(p) and 0 <= p <= 1 for p in self.confidences)
        )
        self.confidence = min(self.confidences) if valid_probs else 0.0
        readings_differ = normalize_plate(self.ocr_plate) != normalize_plate(self.camera_plate)
        # Exact non-empty agreement takes precedence over every warning/threshold.
        # Keep warnings and confidence available for image review nonetheless.
        if self.ocr_plate and self.ocr_plate == self.camera_plate:
            self.status = "SAME"
        # The special clipping rule takes precedence over generic low confidence.
        # A small image alone is not evidence that a character is clipped.
        elif (not self.error and normalize_plate(self.camera_plate) and self.detected_clipping
                and readings_differ and (
                    (valid_probs and self.confidence < .60)
                    or len(self.ocr_plate) != len(self.camera_plate))):
            self.status = "CLIPPED"
        elif (self.error or self.manual_crop or self.crop_reasons or not valid_probs
                or not self.ocr_plate.isalnum() or self.confidence < threshold):
            self.status = "LOWCONFIDENCE"
        else:
            self.status = (
                "DIFF" if readings_differ else "SAME"
            )

    def sheet_row(self) -> list[str]:
        return [self.timestamp, self.camera, self.camera_plate, self.ocr_plate, self.status]

    @property
    def explanation(self) -> str:
        reasons = ([self.error] if self.error else []) + self.crop_reasons
        if self.manual_crop:
            reasons.append("Manually marked overly cropped")
        if not self.confidences:
            reasons.append("No usable character confidence")
        if self.status == "SAME" and self.ocr_plate and self.ocr_plate == self.camera_plate:
            reasons.append("Exact OCR/camera text match overrides confidence and crop/error warnings")
        if self.status == "CLIPPED":
            reasons.append("Clipping detected and OCR reading differs from camera")
            if self.confidences and self.confidence < .60:
                reasons.append("A recognised character has confidence below 60%")
            if len(self.ocr_plate) != len(self.camera_plate):
                reasons.append(f"String lengths differ: OCR {len(self.ocr_plate)}, camera {len(self.camera_plate)}")
        reasons.append(f"Minimum character confidence: {self.confidence:.1%}")
        return "\n".join(reasons)


def crop_reasons(image: Image.Image) -> list[str]:
    """Flag low-resolution crops and character-sized ink touching image boundaries.

    This is a conservative review heuristic, not a trained cropping classifier.
    It can mistake borders/dirt for clipped text and cannot detect text completely
    outside the frame. OCR confidence alone does not solve that ambiguity.
    """
    import cv2

    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    reasons = []
    if height < 20 or width < 60:
        reasons.append(f"Very small plate crop ({width} × {height})")
    if height < 5 or width < 5:
        return reasons
    enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        # Exclude broad background regions, thin plate frames, and tiny specks.
        if not (max(5, height * .25) <= h <= height * .97
                and 2 <= w <= width * .22 and area >= 10
                and .12 <= area / (w * h) <= .95):
            continue
        sides = []
        if x == 0 and np.count_nonzero(labels[y:y+h, 0] == index) / h >= .20:
            sides.append("left")
        if x + w == width and np.count_nonzero(labels[y:y+h, -1] == index) / h >= .20:
            sides.append("right")
        if y == 0 and np.count_nonzero(labels[0, x:x+w] == index) / w >= .25:
            sides.append("top")
        if y + h == height and np.count_nonzero(labels[-1, x:x+w] == index) / w >= .25:
            sides.append("bottom")
        reasons.extend(f"Possible clipped character at {side} edge" for side in sides)
    return list(dict.fromkeys(reasons))


def discover_images(folder: Path, cancelled: Callable[[], bool] = lambda: False) -> list[Path]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Choose an existing folder")
    files = []

    def fail(error: OSError) -> None:
        raise error  # An unreadable subtree must not silently vanish from the report.

    for parent, dirs, names in os.walk(folder, followlinks=False, onerror=fail):
        if cancelled():
            break
        dirs[:] = sorted(d for d in dirs if not (Path(parent) / d).is_symlink())
        for name in sorted(names):
            path = Path(parent) / name
            if path.suffix.lower() in IMAGE_EXTENSIONS and not path.is_symlink():
                files.append(path)
    return files


class OcrEngine:
    def __init__(self, settings: Settings):
        from fast_plate_ocr import LicensePlateRecognizer

        self.settings = settings
        self.recognizer = LicensePlateRecognizer(settings.model, device="cpu")

    def process(self, path: Path) -> Result:
        meta = parse_filename(path, self.settings.filename)
        result = Result(str(path), meta.timestamp, meta.camera, meta.camera_plate, error=meta.error)
        try:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            if self.settings.crop_check:
                result.crop_reasons = crop_reasons(image)
            mode = self.recognizer.config.image_color_mode
            pixels = np.asarray(image.convert("L") if mode == "grayscale" else image)
            prediction = self.recognizer.run(pixels, return_confidence=True)[0]
            result.ocr_plate = prediction.plate
            result.confidences = (
                [float(p) for p in prediction.char_probs[:len(prediction.plate)]]
                if prediction.char_probs is not None else []
            )
        except Exception as exc:
            result.error = "; ".join(filter(None, [result.error, f"Image/OCR error: {exc}"]))
        result.classify(self.settings.threshold)
        return result


@dataclass
class Report:
    folder: str
    settings: Settings
    results: list[Result] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    complete: bool = False
    discovered: int = 0

    @property
    def directory(self) -> Path:
        return data_directory() / "runs" / self.run_id

    def checkpoint(self, result: Result) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "checkpoint.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            stream.flush()

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        atomic_json(self.directory / "report.json", payload)
        self.save_csv(self.directory / "results.csv")
        return self.directory / "report.json"

    def save_csv(self, path: Path) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADERS)
            for result in self.results:
                # Prevent formula injection when the fallback CSV is opened in Excel.
                writer.writerow(["'" + v if v.startswith(("=", "+", "-", "@")) else v
                                 for v in result.sheet_row()])

    @classmethod
    def load(cls, path: Path) -> Report:
        if path.name == "checkpoint.jsonl":
            payload = json.loads((path.parent / "manifest.json").read_text())
            results = []
            for line in path.read_text().splitlines():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # A crash can leave a truncated final checkpoint record.
            payload["results"] = results
            payload["complete"] = False
        else:
            payload = json.loads(path.read_text())
        payload["settings"] = Settings(**payload["settings"])
        payload["results"] = [Result(**row) for row in payload["results"]]
        return cls(**payload)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
