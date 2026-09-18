"""Project-local NudeNet 3.4.2 adapter with explicit ORT providers.

WHY this exists
---------------
Stock NudeNet 3.4.2 ``NudeDetector.__init__`` creates an InferenceSession with the
``providers=`` argument **commented out**. Even when CUDA is available and callers
pass ``providers=...``, the stock detector always stays on CPU.

We must NOT patch site-packages. Instead ``NudeNetBackend``:

1. Locates installed ``320n.onnx`` via ``Path(nudenet.__file__).parent / "320n.onnx"``
   (or an optional user model path — onnx is never committed to git).
2. Creates ``onnxruntime.InferenceSession(model, providers=...)`` itself.
3. GPU mode: call ``ort.preload_dlls(directory="")`` *before* session create,
   then ``["CUDAExecutionProvider", "CPUExecutionProvider"]``. Preload failure
   raises (no silent CPU fallback). Injected ``session=`` skips preload.
4. CPU mode: ``["CPUExecutionProvider"]`` — never calls CUDA preload.
5. After create, inspects ``session.get_providers()``; if GPU was requested and
   ``CUDAExecutionProvider`` is not active → raises (no silent CPU fallback).
6. Reimplements preprocess/postprocess equivalent to NudeNet 3.4.2
   (``_read_image`` / ``_postprocess`` / labels) so we own the session boundary.
7. No long-term global monkeypatch of nudenet.

``detect_batch`` stacks blobs like stock NudeNet when safe; scanner may also call
``detect`` per frame.

Preprocess note (OpenCV BGR video frames)
-----------------------------------------
Stock NudeNet 3.4.2 always runs ``cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)`` before
``blobFromImage(..., swapRB=True)``. On a 3-channel BGR ndarray that conversion
swaps R/B; combined with swapRB the model input matches stock. Treating BGR as-is
then swapRB alone inverts channels relative to stock — FrameSentry must mirror
the stock ``_read_image`` path exactly (only ORT providers differ).
"""

from __future__ import annotations

import time

import _io
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from framesentry.core.logging_setup import get_logger
from framesentry.detectors.base import OrtProviderInfo

logger = get_logger(__name__)

# Exact NudeNet 3.4.2 label order (index = class_id).
NUDENET_LABELS: list[str] = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

GPU_PROVIDERS: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_PROVIDERS: list[str] = ["CPUExecutionProvider"]


class GpuProviderUnavailableError(RuntimeError):
    """Raised when GPU mode was requested but CUDA EP is not active on the session."""


def resolve_default_model_path(model_path: str | Path | None = None) -> Path:
    """Locate 320n.onnx from nudenet package or user path. Does not download."""
    if model_path is not None:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"NudeNet model not found: {p}")
        return p
    import nudenet

    p = Path(nudenet.__file__).resolve().parent / "320n.onnx"
    if not p.is_file():
        raise FileNotFoundError(
            f"Installed nudenet package has no 320n.onnx at {p}. "
            "Install nudenet==3.4.2 (wheel includes the model) or pass model_path."
        )
    return p


def get_ort_provider_info() -> OrtProviderInfo:
    """Query onnxruntime available providers without loading a model."""
    try:
        import onnxruntime as ort
    except ImportError:
        return OrtProviderInfo(
            available_providers=(),
            cuda_listed=False,
            gpu_mode_usable=False,
        )
    providers = tuple(ort.get_available_providers())
    cuda = "CUDAExecutionProvider" in providers
    return OrtProviderInfo(
        available_providers=providers,
        cuda_listed=cuda,
        gpu_mode_usable=cuda,
    )


def select_providers(device: str) -> list[str]:
    """Return provider list for 'gpu' or 'cpu'. Does not create a session."""
    d = (device or "cpu").strip().lower()
    if d == "gpu":
        return list(GPU_PROVIDERS)
    if d == "cpu":
        return list(CPU_PROVIDERS)
    raise ValueError(f"device must be 'gpu' or 'cpu', got {device!r}")


def _read_image_bgr(image: Any, target_size: int = 320) -> tuple:
    """Preprocess matching NudeNet 3.4.2 ``_read_image`` (incl. 3ch BGR path).

    Stock always applies ``cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)`` even for
    3-channel OpenCV BGR frames, then ``blobFromImage(..., swapRB=True)``.
    Skipping that conversion causes RGB/BGR inversion vs stock model input.
    """
    if isinstance(image, str):
        mat = cv2.imread(image)
        if mat is None:
            raise ValueError(f"failed to read image: {image}")
    elif isinstance(image, np.ndarray):
        mat = image
    elif isinstance(image, bytes):
        mat = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    elif isinstance(image, _io.BufferedReader):
        mat = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)
    else:
        raise ValueError("image must be str, np.ndarray, bytes, or BufferedReader")

    if mat is None or mat.size == 0:
        raise ValueError("empty image")

    image_original_width, image_original_height = mat.shape[1], mat.shape[0]

    # Match stock NudeNet 3.4.2 exactly (including 3-channel BGR ndarrays).
    # Grayscale is outside stock's documented path; convert to BGR first so
    # COLOR_RGBA2BGR can run the same channel swap as stock on 3ch mats.
    if mat.ndim == 2:
        mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
    mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

    max_size = max(mat_c3.shape[:2])
    x_pad = max_size - mat_c3.shape[1]
    x_ratio = max_size / mat_c3.shape[1]
    y_pad = max_size - mat_c3.shape[0]
    y_ratio = max_size / mat_c3.shape[0]

    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    input_blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,
        (target_size, target_size),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )

    return (
        input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


def _postprocess(
    output: Any,
    x_pad: float,
    y_pad: float,
    x_ratio: float,
    y_ratio: float,
    image_original_width: int,
    image_original_height: int,
    model_width: int,
    model_height: int,
) -> list[dict[str, Any]]:
    """Postprocess equivalent to NudeNet 3.4.2 ``_postprocess``."""
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = float(np.amax(classes_scores))
        if max_score >= 0.2:
            class_id = int(np.argmax(classes_scores))
            x, y, w, h = outputs[i][0:4]
            x = x - w / 2
            y = y - h / 2
            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height
            x = max(0, min(x, image_original_width))
            y = max(0, min(y, image_original_height))
            w = min(w, image_original_width - x)
            h = min(h, image_original_height - y)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    detections: list[dict[str, Any]] = []
    if indices is None or len(indices) == 0:
        return detections

    for i in indices:
        idx = int(i) if not isinstance(i, (list, tuple, np.ndarray)) else int(i[0])
        box = boxes[idx]
        score = scores[idx]
        class_id = class_ids[idx]
        x, y, w, h = box
        detections.append(
            {
                "class": NUDENET_LABELS[class_id],
                "score": float(score),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )
    return detections


class NudeNetBackend:
    """Owns an onnxruntime session with explicit providers (no nudenet monkeypatch)."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_path: str | Path | None = None,
        inference_resolution: int = 320,
        session: Any | None = None,
    ) -> None:
        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.device = device.strip().lower()
        self.providers = select_providers(self.device)

        if session is not None:
            # Caller-injected session: do not preload CUDA DLLs (no side effects).
            self.session = session
        else:
            import onnxruntime as ort

            # GPU self-built session: preload pip NVIDIA CUDA/cuDNN DLLs first
            # (ORT 1.29+; directory="" = site-packages). CPU must not call this.
            if self.device == "gpu":
                try:
                    ort.preload_dlls(directory="")
                except Exception as exc:  # noqa: BLE001
                    raise GpuProviderUnavailableError(
                        "Failed to preload CUDA/cuDNN DLLs before creating a GPU "
                        f"InferenceSession: {exc}. "
                        "Do not fall back to CPU; fix the GPU install "
                        "(scripts/setup-gpu.ps1) and retry."
                    ) from exc

            model = resolve_default_model_path(model_path)
            self.session = ort.InferenceSession(str(model), providers=self.providers)

        active = list(self.session.get_providers())
        if self.device == "gpu" and "CUDAExecutionProvider" not in active:
            raise GpuProviderUnavailableError(
                "GPU mode requested but CUDAExecutionProvider is not active on the "
                f"session. Active providers: {active}. "
                "Install onnxruntime-gpu (and uninstall plain onnxruntime), ensure "
                "NVIDIA drivers/CUDA/cuDNN match the ORT build, then retry."
            )

        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.last_batch_timing: dict[str, float] = {
            "preprocess_tensor_sec": 0.0,
            "inference_sec": 0.0,
            "postprocess_sec": 0.0,
        }
        logger.info(
            "NudeNetBackend ready device=%s requested=%s active=%s",
            self.device,
            self.providers,
            active,
        )

    def onnx_input_batch_dim(self) -> object:
        """Return ONNX input batch dimension (str/None = dynamic, int = fixed)."""
        try:
            shape = self.session.get_inputs()[0].shape
            return shape[0] if shape else None
        except Exception:  # noqa: BLE001
            return None

    def supports_batch_gt1(self) -> bool:
        """True when ONNX input batch dim is dynamic or >1 (real batching)."""
        dim = self.onnx_input_batch_dim()
        if dim is None:
            return True
        if isinstance(dim, str):
            return True
        try:
            return int(dim) != 1
        except (TypeError, ValueError):
            return True


    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        (
            preprocessed_image,
            x_ratio,
            y_ratio,
            x_pad,
            y_pad,
            image_original_width,
            image_original_height,
        ) = _read_image_bgr(image_bgr, self.input_width)
        outputs = self.session.run(None, {self.input_name: preprocessed_image})
        return _postprocess(
            outputs,
            x_pad,
            y_pad,
            x_ratio,
            y_ratio,
            image_original_width,
            image_original_height,
            self.input_width,
            self.input_height,
        )

    def detect_batch(
        self, images_bgr: list[np.ndarray], batch_size: int = 4
    ) -> list[list[dict[str, Any]]]:
        """Batch detect like stock NudeNet (vstack blobs). Safe for equal-size pads.

        Timing (summed across mini-batches) is stored on ``self.last_batch_timing``:
        - preprocess_tensor_sec: ``_read_image_bgr`` + vstack only
        - inference_sec: ONLY ``session.run(...)``
        - postprocess_sec: NudeNet ``_postprocess`` only (not target filter / JPEG)
        """
        all_detections: list[list[dict[str, Any]]] = []
        t_preprocess = 0.0
        t_inference = 0.0
        t_postprocess = 0.0
        for i in range(0, len(images_bgr), batch_size):
            batch = images_bgr[i : i + batch_size]
            batch_inputs: list[np.ndarray] = []
            batch_metadata: list[tuple] = []
            t0 = time.perf_counter()
            for image in batch:
                (
                    preprocessed_image,
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = _read_image_bgr(image, self.input_width)
                batch_inputs.append(preprocessed_image)
                batch_metadata.append(
                    (
                        x_ratio,
                        y_ratio,
                        x_pad,
                        y_pad,
                        image_original_width,
                        image_original_height,
                    )
                )
            batch_input = np.vstack(batch_inputs)
            t_preprocess += time.perf_counter() - t0
            t1 = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: batch_input})
            t_inference += time.perf_counter() - t1
            t2 = time.perf_counter()
            for j, metadata in enumerate(batch_metadata):
                (
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = metadata
                detections = _postprocess(
                    [outputs[0][j : j + 1]],
                    x_pad,
                    y_pad,
                    x_ratio,
                    y_ratio,
                    image_original_width,
                    image_original_height,
                    self.input_width,
                    self.input_height,
                )
                all_detections.append(detections)
            t_postprocess += time.perf_counter() - t2
        self.last_batch_timing = {
            "preprocess_tensor_sec": t_preprocess,
            "inference_sec": t_inference,
            "postprocess_sec": t_postprocess,
        }
        return all_detections


def create_nudenet_backend(
    *,
    device: str = "cpu",
    model_path: str | Path | None = None,
) -> NudeNetBackend:
    """Factory used by scanner/GUI (GUI never imports nudenet)."""
    logger.info("GPU/detector create begin device=%s", device)
    try:
        backend = NudeNetBackend(device=device, model_path=model_path)
    except Exception:
        logger.exception("GPU/detector create error device=%s", device)
        raise
    logger.info("GPU/detector create end device=%s", device)
    return backend
