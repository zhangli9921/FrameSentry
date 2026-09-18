"""Provider selection, preload_dlls, and GPU-requested-but-CPU-session fails."""

from unittest.mock import MagicMock, patch

import pytest

from framesentry.detectors.nudenet_backend import (
    GPU_PROVIDERS,
    CPU_PROVIDERS,
    GpuProviderUnavailableError,
    NudeNetBackend,
    select_providers,
    get_ort_provider_info,
)


def test_select_providers():
    assert select_providers("gpu") == GPU_PROVIDERS
    assert select_providers("cpu") == CPU_PROVIDERS
    with pytest.raises(ValueError):
        select_providers("tpu")


def test_gpu_requested_cpu_session_raises(mock_inference_session):
    mock_inference_session.get_providers.return_value = ["CPUExecutionProvider"]
    with pytest.raises(GpuProviderUnavailableError):
        NudeNetBackend(device="gpu", session=mock_inference_session)


def test_cpu_session_ok(mock_inference_session):
    mock_inference_session.get_providers.return_value = ["CPUExecutionProvider"]
    backend = NudeNetBackend(device="cpu", session=mock_inference_session)
    assert backend.input_name == "images"


def test_gpu_session_ok_when_cuda_active(mock_inference_session):
    mock_inference_session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    backend = NudeNetBackend(device="gpu", session=mock_inference_session)
    assert backend.device == "gpu"


def test_get_ort_provider_info_no_crash():
    info = get_ort_provider_info()
    assert isinstance(info.available_providers, tuple)
    # Never claim CUDA if not listed
    if not info.cuda_listed:
        assert info.gpu_mode_usable is False


def test_gpu_self_built_session_calls_preload_dlls(tmp_path):
    """GPU path that creates its own session must preload CUDA DLLs first."""
    model = tmp_path / "320n.onnx"
    model.write_bytes(b"fake-onnx")

    mock_session = MagicMock()
    mock_session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    inp = MagicMock()
    inp.name = "images"
    mock_session.get_inputs.return_value = [inp]

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    mock_ort.preload_dlls = MagicMock()

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        with patch(
            "framesentry.detectors.nudenet_backend.resolve_default_model_path",
            return_value=model,
        ):
            backend = NudeNetBackend(device="gpu", model_path=model)

    mock_ort.preload_dlls.assert_called_once_with(directory="")
    mock_ort.InferenceSession.assert_called_once()
    assert backend.device == "gpu"


def test_cpu_self_built_session_does_not_preload(tmp_path):
    """CPU mode must not call ort.preload_dlls (no CUDA DLL dependency)."""
    model = tmp_path / "320n.onnx"
    model.write_bytes(b"fake-onnx")

    mock_session = MagicMock()
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    inp = MagicMock()
    inp.name = "images"
    mock_session.get_inputs.return_value = [inp]

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    mock_ort.preload_dlls = MagicMock()

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        with patch(
            "framesentry.detectors.nudenet_backend.resolve_default_model_path",
            return_value=model,
        ):
            NudeNetBackend(device="cpu", model_path=model)

    mock_ort.preload_dlls.assert_not_called()


def test_injected_session_does_not_preload(mock_inference_session):
    """Caller-injected session must not force CUDA preload side effects."""
    mock_inference_session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    mock_ort = MagicMock()
    mock_ort.preload_dlls = MagicMock()

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        NudeNetBackend(device="gpu", session=mock_inference_session)

    mock_ort.preload_dlls.assert_not_called()


def test_preload_failure_hard_fails_no_cpu_fallback(tmp_path):
    """If preload_dlls fails on GPU path, raise — do not create CPU session."""
    model = tmp_path / "320n.onnx"
    model.write_bytes(b"fake-onnx")

    mock_ort = MagicMock()
    mock_ort.preload_dlls.side_effect = OSError("cudnn not found")
    mock_ort.InferenceSession = MagicMock()

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        with patch(
            "framesentry.detectors.nudenet_backend.resolve_default_model_path",
            return_value=model,
        ):
            with pytest.raises(GpuProviderUnavailableError, match="preload"):
                NudeNetBackend(device="gpu", model_path=model)

    mock_ort.InferenceSession.assert_not_called()


def test_gpu_final_providers_without_cuda_hard_fail(tmp_path):
    """GPU requested but final session providers lack CUDA → hard fail."""
    model = tmp_path / "320n.onnx"
    model.write_bytes(b"fake-onnx")

    mock_session = MagicMock()
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    inp = MagicMock()
    inp.name = "images"
    mock_session.get_inputs.return_value = [inp]

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    mock_ort.preload_dlls = MagicMock()

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        with patch(
            "framesentry.detectors.nudenet_backend.resolve_default_model_path",
            return_value=model,
        ):
            with pytest.raises(GpuProviderUnavailableError):
                NudeNetBackend(device="gpu", model_path=model)
