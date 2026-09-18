"""Provider selection and GPU-requested-but-CPU-session fails."""

from unittest.mock import MagicMock

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
