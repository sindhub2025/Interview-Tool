"""Regression checks for runtime data required by the frozen executable."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

from ghostmic.main import get_build_source_signature


def test_spec_collects_only_faster_whisper_vad_onnx_asset():
    spec_text = Path("InterviewTool.spec").read_text(encoding="utf-8")
    collected = collect_data_files("faster_whisper", includes=["assets/*.onnx"])

    assert 'includes=["assets/*.onnx"]' in spec_text
    assert any(
        Path(source).name == "silero_vad_v6.onnx"
        and destination.replace("\\", "/") == "faster_whisper/assets"
        for source, destination in collected
    )


def test_build_source_signature_is_deterministic():
    first = get_build_source_signature()
    second = get_build_source_signature()

    assert first == second
    assert len(first) == 16