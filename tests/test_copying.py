import json
from pathlib import Path

import pytest

from plate_ocr_review import copying
from plate_ocr_review.core import Report, Result, Settings


@pytest.fixture
def report(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    source = tmp_path / "source"
    source.mkdir()
    return Report(str(source), Settings(), complete=True)


def add_image(report, name, status="DIFF", plate="ABC123", content=b"image bytes"):
    path = Path(report.folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    report.results.append(Result(str(path), ocr_plate=plate, status=status))
    return path


def destination(tmp_path):
    folder = tmp_path / "copies"
    folder.mkdir()
    return folder


def test_only_diff_copied_with_literal_suffix(report, tmp_path):
    source = add_image(report, "subtree/photo.JPG")
    for status in ["SAME", "CLIPPED", "LOWCONFIDENCE"]:
        add_image(report, status + ".jpg", status=status)
    target = destination(tmp_path)
    summary = copying.copy_diff_images(report, target)
    assert summary.requested == summary.copied == 1
    assert [p.name for p in target.iterdir()] == ["photo_<ABC123>.JPG"]
    assert (target / "photo_<ABC123>.JPG").read_bytes() == source.read_bytes() == b"image bytes"
    assert source.exists()
    assert json.loads(Path(summary.receipt).read_text())["records"][0]["outcome"] == "copied"


def test_conflicts_never_overwritten_and_duplicate_names_supported(report, tmp_path):
    add_image(report, "one/photo.jpg", content=b"first")
    add_image(report, "two/photo.jpg", content=b"second")
    target = destination(tmp_path)
    existing = target / "photo_<ABC123>.jpg"
    existing.write_bytes(b"unrelated")
    summary = copying.copy_diff_images(report, target)
    assert summary.copied == 2
    assert existing.read_bytes() == b"unrelated"
    assert (target / "photo_2_<ABC123>.jpg").read_bytes() == b"first"
    assert (target / "photo_3_<ABC123>.jpg").read_bytes() == b"second"


def test_repeat_copy_skips_identical_files(report, tmp_path):
    add_image(report, "photo.jpg")
    target = destination(tmp_path)
    copying.copy_diff_images(report, target)
    summary = copying.copy_diff_images(report, target)
    assert summary.copied == 0
    assert summary.already_present == 1
    assert len(list(target.iterdir())) == 1


def test_changed_source_gets_new_copy_without_overwriting_prior_copy(report, tmp_path):
    source = add_image(report, "photo.jpg", content=b"before")
    target = destination(tmp_path)
    copying.copy_diff_images(report, target)
    source.write_bytes(b"after!")
    summary = copying.copy_diff_images(report, target)
    assert summary.copied == 1
    assert (target / "photo_<ABC123>.jpg").read_bytes() == b"before"
    assert (target / "photo_2_<ABC123>.jpg").read_bytes() == b"after!"


def test_missing_file_does_not_stop_remaining_copies(report, tmp_path):
    missing = add_image(report, "missing.jpg")
    missing.unlink()
    add_image(report, "okay.jpg")
    summary = copying.copy_diff_images(report, destination(tmp_path))
    assert summary.requested == 2
    assert summary.copied == 1
    assert len(summary.errors) == 1


@pytest.mark.parametrize("plate", ["", "../escape", "AB/CD", "AB\\CD", "AB>CD", "AB\x00CD"])
def test_unsafe_ocr_text_cannot_escape_destination(report, tmp_path, plate):
    add_image(report, "photo.jpg", plate=plate)
    target = destination(tmp_path)
    summary = copying.copy_diff_images(report, target)
    assert summary.copied == 0
    assert len(summary.errors) == 1
    assert not list(target.iterdir())


def test_symlinks_not_followed_or_overwritten(report, tmp_path):
    original = add_image(report, "original.jpg")
    alias = Path(report.folder) / "alias.jpg"
    alias.symlink_to(original)
    report.results.append(Result(str(alias), ocr_plate="ABC123", status="DIFF"))
    target = destination(tmp_path)
    occupied = target / "original_<ABC123>.jpg"
    occupied.symlink_to(tmp_path / "unrelated-missing.jpg")
    summary = copying.copy_diff_images(report, target)
    assert summary.copied == 1
    assert len(summary.errors) == 1
    assert occupied.is_symlink()
    assert (target / "original_2_<ABC123>.jpg").exists()


def test_cancel_retains_completed_copies(report, tmp_path):
    for i in range(3):
        add_image(report, f"{i}.jpg")
    progress = []
    summary = copying.copy_diff_images(report, destination(tmp_path),
        progress=lambda done, total: progress.append((done, total)), cancelled=lambda: bool(progress))
    assert summary.cancelled
    assert summary.copied == 1
    assert progress == [(1, 3)]


def test_partial_copy_removed_and_next_file_processed(report, tmp_path, monkeypatch):
    add_image(report, "one.jpg")
    add_image(report, "two.jpg")
    target = destination(tmp_path)
    original = copying.shutil.copyfileobj
    calls = []

    def fail_once(source, outgoing, length):
        calls.append(True)
        if len(calls) == 1:
            outgoing.write(b"partial")
            raise OSError("Simulated disk error")
        return original(source, outgoing, length)

    monkeypatch.setattr(copying.shutil, "copyfileobj", fail_once)
    summary = copying.copy_diff_images(report, target)
    assert summary.copied == 1
    assert len(summary.errors) == 1
    assert not (target / "one_<ABC123>.jpg").exists()
    assert (target / "two_<ABC123>.jpg").read_bytes() == b"image bytes"


def test_invalid_destination_or_no_diffs(report, tmp_path):
    with pytest.raises(ValueError, match="destination"):
        copying.copy_diff_images(report, tmp_path / "missing")
    with pytest.raises(ValueError, match="no DIFF"):
        copying.copy_diff_images(report, destination(tmp_path))
