"""Copy DIFF images without modifying sources or overwriting destination files."""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .core import Report, atomic_json


@dataclass
class CopySummary:
    destination: str
    requested: int
    copied: int = 0
    already_present: int = 0
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False
    records: list[dict] = field(default_factory=list)
    receipt: str = ""

    @property
    def processed(self) -> int:
        return self.copied + self.already_present + len(self.errors)


def identical_files(source: Path, target: Path) -> bool:
    """Compare bytes afresh; cached mtime-based comparisons can become stale."""
    try:
        with source.open("rb") as left, target.open("rb") as right:
            if os.fstat(left.fileno()).st_size != os.fstat(right.fileno()).st_size:
                return False
            while True:
                left_bytes, right_bytes = left.read(1024 * 1024), right.read(1024 * 1024)
                if left_bytes != right_bytes:
                    return False
                if not left_bytes:
                    return True
    except OSError:
        return False


def copy_one(source: Path, destination: Path, plate: str) -> tuple[Path, bool]:
    if not plate or not plate.isalnum():
        raise ValueError("OCR reading must contain only letters and digits for a copy filename")
    if source.is_symlink():
        raise ValueError("Source is now a symbolic link; not copied")
    if not source.is_file():
        raise ValueError("Source image is missing or is not a regular file")
    with source.open("rb") as incoming:
        if not stat.S_ISREG(os.fstat(incoming.fileno()).st_mode):
            raise ValueError("Source is not a regular image file")
        counter = 1
        while True:
            disambiguation = "" if counter == 1 else f"_{counter}"
            target = destination / f"{source.stem}{disambiguation}_<{plate}>{source.suffix}"
            try:
                outgoing = target.open("xb")  # Exclusive creation: never truncate an existing file.
            except FileExistsError:
                if not target.is_symlink() and target.is_file() and identical_files(source, target):
                    return target, False
                counter += 1
                continue
            try:
                with outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            except BaseException:
                # This operation created this exact file; remove only its partial copy.
                target.unlink(missing_ok=True)
                raise
            return target, True


def copy_diff_images(
    report: Report,
    destination: Path,
    progress: Callable[[int, int], None] = lambda _done, _total: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> CopySummary:
    if os.name == "nt":
        raise ValueError("Windows filenames cannot contain < or >; the requested suffix requires a compatible filesystem")
    destination = destination.expanduser().resolve()
    if not destination.is_dir():
        raise ValueError("Choose an existing destination folder")
    rows = [result for result in report.results if result.status == "DIFF"]
    if not rows:
        raise ValueError("There are no DIFF images to copy")
    summary = CopySummary(str(destination), len(rows))
    for result in rows:
        if cancelled():
            summary.cancelled = True
            break
        record = {"source": result.path, "ocr_plate": result.ocr_plate}
        try:
            target, copied = copy_one(Path(result.path), destination, result.ocr_plate)
            if copied:
                summary.copied += 1
            else:
                summary.already_present += 1
            record.update(destination=str(target), outcome="copied" if copied else "already present")
        except Exception as exc:
            summary.errors.append(f"{result.path}: {exc}")
            record.update(outcome="error", error=str(exc))
        summary.records.append(record)
        progress(summary.processed, summary.requested)
    name = datetime.now().strftime("copy-diff-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8] + ".json"
    summary.receipt = str(report.directory / name)
    try:
        atomic_json(Path(summary.receipt), asdict(summary))
    except OSError as exc:
        raise RuntimeError(
            f"Copied {summary.copied} images, but could not save the local copy log: {exc}. "
            "Existing copies are safe; copying again skips identical files."
        ) from exc
    return summary
