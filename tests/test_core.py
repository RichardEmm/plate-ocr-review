from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from plate_ocr_review.core import (
    FilenameConfig, OcrEngine, Report, Result, Settings, crop_reasons, discover_images,
    normalize_plate, parse_filename,
)


@pytest.mark.parametrize("name,timestamp,camera,plate", [
    ("26-09-08_17-06-27_DOVER24_GSV81G_ABC.jpg", "2026-09-08 17:06:27", "DOVER24", "GSV81G"),
    ("26-09-08_17-06-27_DOVER2_00123.jpg", "2026-09-08 17:06:27", "DOVER2", "00123"),
    ("26-09-08_17-06-27_dover24_gsv81g_ABC.JPG", "2026-09-08 17:06:27", "dover24", "gsv81g"),
    ("26-09-08_17-06-27_GateEast_GSV81G_ABC.jpg", "2026-09-08 17:06:27", "GateEast", "GSV81G"),
    ("26-09-08_17-06-27_DOVER24_ABC_X_Y.jpg", "2026-09-08 17:06:27", "DOVER24", "ABC"),
])
def test_filename(name, timestamp, camera, plate):
    result = parse_filename(Path(name))
    assert (result.timestamp, result.camera, result.camera_plate, result.error) == (timestamp, camera, plate, "")


@pytest.mark.parametrize("name", [
    "random.jpg", "26-02-30_17-06-27_DOVER24_ABC_X.jpg",
    "26-09-08_25-06-27_DOVER24_ABC_X.jpg",
    "26-09-08_17-06-27_DOVER24__X.jpg",
    "26-09-08_17-06-27__ABC_X.jpg",
])
def test_bad_filename_not_silently_accepted(name):
    assert parse_filename(Path(name)).error


@pytest.mark.parametrize("name,config,camera,plate", [
    ("26-09-08_17-06-27_GSV81G_DOVER24_ABC.jpg",
     FilenameConfig(camera_field=4, plate_field=3), "DOVER24", "GSV81G"),
    ("26-09-08_17-06-27_other_GateEast_GSV81G_ABC.jpg",
     FilenameConfig(camera_field=-3, plate_field=-2), "GateEast", "GSV81G"),
    ("26-09-08_17-06-27_other_GateEast_GSV81G.jpg",
     FilenameConfig(camera_field=-2, plate_field=-1), "GateEast", "GSV81G"),
    ("GateEast_GSV81G_26-09-08_17-06-27.jpg",
     FilenameConfig(camera_field=1, plate_field=2, date_field=3, time_field=4), "GateEast", "GSV81G"),
    ("26-09-08;17-06-27;GateEast;00123;suffix.jpg",
     FilenameConfig(separator=";"), "GateEast", "00123"),
    ("26-09-08__17-06-27__GateEast__00123__suffix.jpg",
     FilenameConfig(separator="__"), "GateEast", "00123"),
])
def test_configurable_filename_fields(name, config, camera, plate):
    result = parse_filename(Path(name), config)
    assert not result.error
    assert (result.camera, result.camera_plate, result.timestamp) == (camera, plate, "2026-09-08 17:06:27")


@pytest.mark.parametrize("config", [
    {"separator": ""}, {"separator": "a" * 17}, {"camera_field": 0},
    {"plate_field": 101}, {"plate_field": -101}, {"plate_field": 3},
    {"camera_field": "3"}, {"camera_field": True},
])
def test_invalid_filename_configuration(config):
    with pytest.raises(ValueError):
        FilenameConfig(**config)


def test_out_of_range_and_relative_alias_fields():
    path = Path("26-09-08_17-06-27_GateEast_ABC_X.jpg")
    assert "Missing camera field" in parse_filename(path, FilenameConfig(camera_field=7)).error
    assert "same filename field" in parse_filename(path, FilenameConfig(plate_field=-3)).error


def test_config_dict_and_report_compatibility(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    settings = Settings(filename={"camera_field": -3, "plate_field": -2})
    assert isinstance(settings.filename, FilenameConfig)
    report = Report("images", settings)
    saved = report.save()
    assert Report.load(saved).settings.filename == settings.filename
    payload = json.loads(saved.read_text())
    del payload["settings"]["filename"]
    saved.write_text(json.dumps(payload))
    assert Report.load(saved).settings.filename == FilenameConfig()


def test_comparison_normalization():
    assert normalize_plate(" ab-12 cd ") == "AB12CD"
    assert normalize_plate("O01I") != normalize_plate("0011")


@pytest.mark.parametrize("plate,prob,camera,status", [
    ("GSV81G", .95, "GSV81G", "SAME"),
    ("GSV81G", .95, "GSV81C", "DIFF"),
    ("GSV81G", .89, "GSV81G", "SAME"),
    ("GSV81G", .89, "GSV81C", "LOWCONFIDENCE"),
    ("GSV81G", .90, "gsv-81g", "SAME"),
    ("", .99, "ABC", "LOWCONFIDENCE"),
    ("AB_C", .99, "ABC", "LOWCONFIDENCE"),
])
def test_status_rules(plate, prob, camera, status):
    result = Result("image.jpg", camera_plate=camera, ocr_plate=plate, confidences=[prob] * len(plate))
    result.classify(.9)
    assert result.status == status


@pytest.mark.parametrize("field,value", [
    ("crop_reasons", ["Clipped edge"]), ("error", "Unreadable filename"),
    ("confidences", []), ("confidences", [.999]),
    ("confidences", [float("nan")] * 3), ("confidences", [1.1] * 3),
])
def test_unsafe_results_override_high_confidence(field, value):
    result = Result("image.jpg", camera_plate="ABC", ocr_plate="ABD", confidences=[.999] * 3)
    setattr(result, field, value)
    result.classify(.9)
    assert result.status == "LOWCONFIDENCE"


def test_weakest_character_not_mean():
    result = Result("image.jpg", camera_plate="ABC", ocr_plate="ABD", confidences=[1, 1, .8])
    result.classify(.9)
    assert result.confidence == .8
    assert result.status == "LOWCONFIDENCE"


@pytest.mark.parametrize("ocr,camera,probs,clipped,status", [
    ("ABD", "ABC", [.99, .99, .59], True, "CLIPPED"),
    ("ABD", "ABC", [.99, .99, .60], True, "CLIPPED"),
    ("ABD", "ABC", [.99, .99, .61], True, "CLIPPED"),
    ("ABD", "ABC", [.99, .99, .75], True, "CLIPPED"),
    ("ABD", "ABC", [.99, .99, .99], True, "CLIPPED"),
    ("AB", "ABC", [.99, .99], True, "CLIPPED"),
    ("ABCD", "ABC", [.99] * 4, True, "CLIPPED"),
    ("AB", "ABC", [], True, "CLIPPED"),
    ("", "ABC", [], True, "CLIPPED"),
    ("ABC", "ABC", [.50] * 3, True, "SAME"),
    ("ABC", "abc", [.50] * 3, True, "LOWCONFIDENCE"),
    ("ABC", "AB-C", [.99] * 3, True, "LOWCONFIDENCE"),
    ("ABD", "AB-C", [.99] * 3, True, "CLIPPED"),
    ("ABD", "ABC", [], True, "CLIPPED"),
    ("ABD", "ABC", [float("nan")] * 3, True, "CLIPPED"),
    ("ABD", "ABC", [.50] * 3, False, "LOWCONFIDENCE"),
    ("AB", "ABC", [.99] * 2, False, "DIFF"),
    ("ABD", "ABC", [.99] * 3, False, "DIFF"),
])
def test_clipped_rule(ocr, camera, probs, clipped, status):
    result = Result("image.jpg", camera_plate=camera, ocr_plate=ocr, confidences=probs,
                    crop_reasons=["Possible clipped character at left edge"] if clipped else [])
    result.classify(.90)
    assert result.status == status


@pytest.mark.parametrize("threshold", [.50, .90, 1.0])
def test_manual_clipping_independent_of_confidence_threshold(threshold):
    result = Result("image.jpg", camera_plate="ABC", ocr_plate="ABD", confidences=[.99, .99, .75],
                    manual_crop=True)
    result.classify(threshold)
    assert result.status == "CLIPPED"
    assert "Confidence and string length do not restrict" in result.explanation
    result.manual_crop = False
    result.classify(.50)
    assert result.status == "DIFF"


def test_small_image_is_not_clipping_evidence():
    result = Result("image.jpg", camera_plate="ABC", ocr_plate="AB", confidences=[.50, .50],
                    crop_reasons=["Very small plate crop (40 × 10)"])
    result.classify(.90)
    assert result.status == "LOWCONFIDENCE"


def test_invalid_data_does_not_become_clipped():
    result = Result("image.jpg", camera_plate="ABC", ocr_plate="AB", confidences=[.50, .50],
                    manual_crop=True, error="Invalid capture date/time")
    result.classify(.90)
    assert result.status == "LOWCONFIDENCE"
    result.error = ""
    result.camera_plate = ""
    result.classify(.90)
    assert result.status == "LOWCONFIDENCE"


@pytest.mark.parametrize("changes", [
    {"confidences": [.01, .99, .99]}, {"confidences": []}, {"confidences": [.99]},
    {"confidences": [float("nan")] * 3}, {"confidences": [1.1] * 3},
    {"manual_crop": True}, {"crop_reasons": ["Possible clipped character at left edge"]},
    {"crop_reasons": ["Very small plate crop (40 × 10)"]},
    {"error": "Invalid capture date/time"},
    {"manual_crop": True, "confidences": [.10] * 3, "error": "Filename warning"},
])
def test_exact_match_always_same(changes):
    result = Result("image.jpg", camera_plate="ABC", ocr_plate="ABC", confidences=[.99] * 3)
    for name, value in changes.items():
        setattr(result, name, value)
    result.classify(.90)
    assert result.status == "SAME"
    assert "Exact OCR/camera text match" in result.explanation


def test_two_empty_readings_are_not_an_exact_plate_match():
    result = Result("image.jpg", camera_plate="", ocr_plate="")
    result.classify(.90)
    assert result.status == "LOWCONFIDENCE"


def test_normalized_match_does_not_get_exact_match_override():
    result = Result("image.jpg", camera_plate="abc", ocr_plate="ABC", confidences=[.10] * 3)
    result.classify(.90)
    assert result.status == "LOWCONFIDENCE"


def test_recursive_images_and_symlinks(tmp_path):
    child = tmp_path / "one" / "two"
    child.mkdir(parents=True)
    (child / "a.JPG").touch()
    (tmp_path / "b.webp").touch()
    (child / "ignore.txt").touch()
    (child / "loop").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "alias.jpg").symlink_to(child / "a.JPG")
    assert set(discover_images(tmp_path)) == {child / "a.JPG", tmp_path / "b.webp"}


def test_no_folder(tmp_path):
    with pytest.raises(ValueError):
        discover_images(tmp_path / "missing")


def test_crop_small_and_plain():
    assert crop_reasons(Image.new("RGB", (40, 10), "white"))
    assert not crop_reasons(Image.new("RGB", (240, 60), "white"))


def test_crop_edge_glyph():
    image = Image.new("RGB", (240, 60), "white")
    draw = ImageDraw.Draw(image)
    # Half an E, with the vertical stroke clipped just outside the image.
    for y in [14, 29, 44]:
        draw.rectangle((0, y, 14, y + 3), fill="black")
    draw.rectangle((0, 14, 3, 47), fill="black")
    assert any("left" in reason for reason in crop_reasons(image))


def fake_engine(settings, predictions):
    engine = OcrEngine.__new__(OcrEngine)
    engine.settings = settings
    engine.recognizer = SimpleNamespace(
        config=SimpleNamespace(image_color_mode="rgb"),
        run=lambda image, return_confidence: predictions,
    )
    return engine


def test_ocr_adapter_ignores_padding_confidence(tmp_path):
    path = tmp_path / "26-09-08_17-06-27_DOVER24_ABC_X.png"
    Image.new("RGB", (240, 60), "white").save(path)
    engine = fake_engine(Settings(), [SimpleNamespace(plate="ABC", char_probs=np.array([.98, .99, .99, .01, .01]))])
    result = engine.process(path)
    assert result.status == "SAME"
    assert result.confidences == [.98, .99, .99]


def test_unreadable_image_has_a_row(tmp_path):
    path = tmp_path / "26-09-08_17-06-27_DOVER24_ABC_X.jpg"
    path.write_text("not an image")
    result = fake_engine(Settings(), []).process(path)
    assert result.status == "LOWCONFIDENCE"
    assert "Image/OCR error" in result.error
    assert result.camera_plate == "ABC"


def test_ocr_uses_configured_camera_and_plate_fields(tmp_path):
    path = tmp_path / "26-09-08_17-06-27_ABC_GateEast_X.png"
    Image.new("RGB", (240, 60), "white").save(path)
    settings = Settings(filename=FilenameConfig(camera_field=4, plate_field=3))
    engine = fake_engine(settings, [SimpleNamespace(plate="ABC", char_probs=np.array([.99] * 5))])
    result = engine.process(path)
    assert (result.camera, result.camera_plate, result.status) == ("GateEast", "ABC", "SAME")


def test_report_roundtrip_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    report = Report("images", Settings())
    row = Result("file", camera_plate="0012", ocr_plate="0012", confidences=[.99] * 4)
    row.classify(.9)
    report.results.append(row)
    saved = report.save()
    loaded = Report.load(saved)
    assert loaded.results[0].sheet_row() == ["", "", "0012", "0012", "SAME"]
    report.checkpoint(row)
    from dataclasses import asdict
    from plate_ocr_review.core import atomic_json
    payload = asdict(report)
    payload["results"] = []
    atomic_json(report.directory / "manifest.json", payload)
    with (report.directory / "checkpoint.jsonl").open("a") as stream:
        stream.write('{"truncated":')
    recovered = Report.load(report.directory / "checkpoint.jsonl")
    assert len(recovered.results) == 1
    assert not recovered.complete


@pytest.mark.parametrize("threshold", [0, -1, 1.1])
def test_threshold_validation(threshold):
    with pytest.raises(ValueError):
        Settings(threshold=threshold)
