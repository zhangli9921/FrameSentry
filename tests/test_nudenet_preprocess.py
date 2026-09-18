"""Regression: FrameSentry preprocess must match stock NudeNet 3.4.2 exactly."""

from __future__ import annotations

import importlib
import inspect

import cv2
import numpy as np

from framesentry.detectors.nudenet_backend import (
    NUDENET_LABELS,
    _postprocess,
    _read_image_bgr,
)


def _stock_nudenet():
    return importlib.import_module("nudenet.nudenet")


def test_labels_match_stock_nudenet_342():
    stock = _stock_nudenet()
    assert NUDENET_LABELS == list(stock.__labels)
    src = inspect.getsource(stock)
    assert "FEMALE_BREAST_EXPOSED" in src
    assert "BUTTOCKS_EXPOSED" in src


def test_preprocess_blob_matches_stock_for_bgr_ndarray():
    """Distinct R/G/B channel values — must catch RGB/BGR inversion."""
    stock = _stock_nudenet()
    # OpenCV BGR layout: B=30, G=20, R=10 (distinct so swap is detectable).
    bgr = np.zeros((40, 60, 3), dtype=np.uint8)
    bgr[:, :, 0] = 30  # B
    bgr[:, :, 1] = 20  # G
    bgr[:, :, 2] = 10  # R

    blob_stock, x_ratio_s, y_ratio_s, x_pad_s, y_pad_s, ow_s, oh_s = stock._read_image(
        bgr, 320
    )
    blob_fs, x_ratio, y_ratio, x_pad, y_pad, ow, oh = _read_image_bgr(bgr, 320)

    assert blob_fs.shape == blob_stock.shape == (1, 3, 320, 320)
    assert np.allclose(blob_fs, blob_stock), (
        "FrameSentry preprocess blob must equal stock NudeNet 3.4.2 _read_image blob"
    )
    assert (x_ratio, y_ratio, x_pad, y_pad, ow, oh) == (
        x_ratio_s,
        y_ratio_s,
        x_pad_s,
        y_pad_s,
        ow_s,
        oh_s,
    )

    # Explicit RGB/BGR inversion guard: wrong path (BGR as-is + swapRB) differs.
    max_size = max(bgr.shape[:2])
    x_pad_w = max_size - bgr.shape[1]
    y_pad_w = max_size - bgr.shape[0]
    mat_pad = cv2.copyMakeBorder(bgr, 0, y_pad_w, 0, x_pad_w, cv2.BORDER_CONSTANT)
    blob_wrong = cv2.dnn.blobFromImage(
        mat_pad, 1 / 255.0, (320, 320), (0, 0, 0), swapRB=True, crop=False
    )
    assert not np.allclose(blob_wrong, blob_stock), "test must detect RGB/BGR inversion"
    assert not np.allclose(blob_wrong, blob_fs)


def test_preprocess_padding_resize_normalization_align_with_stock():
    stock = _stock_nudenet()
    bgr = np.zeros((45, 80, 3), dtype=np.uint8)
    bgr[:, :, 0] = 5
    bgr[:, :, 1] = 15
    bgr[:, :, 2] = 25
    blob_s, *meta_s = stock._read_image(bgr, 320)
    blob_f, *meta_f = _read_image_bgr(bgr, 320)
    assert meta_f == meta_s
    assert np.allclose(blob_f, blob_s)
    assert float(blob_f.max()) <= 1.0 + 1e-5
    assert float(blob_f.min()) >= -1e-5


def test_stock_source_uses_rgba2bgr_then_swaprb():
    stock = _stock_nudenet()
    src = inspect.getsource(stock._read_image)
    assert "COLOR_RGBA2BGR" in src
    assert "swapRB=True" in src or "swapRB = True" in src


def test_postprocess_aligns_with_stock_empty_and_boxes():
    stock = _stock_nudenet()
    nc = len(NUDENET_LABELS)
    raw = np.zeros((1, 4 + nc, 2), dtype=np.float32)
    raw[0, 0, 0] = 160  # cx
    raw[0, 1, 0] = 160  # cy
    raw[0, 2, 0] = 40  # w
    raw[0, 3, 0] = 40  # h
    raw[0, 4 + 3, 0] = 0.9  # FEMALE_BREAST_EXPOSED

    kwargs = dict(
        x_pad=0,
        y_pad=0,
        x_ratio=1.0,
        y_ratio=1.0,
        image_original_width=320,
        image_original_height=320,
        model_width=320,
        model_height=320,
    )
    det_fs = _postprocess([raw], **kwargs)
    det_stock = stock._postprocess([raw], **kwargs)
    assert len(det_fs) == len(det_stock) >= 1
    assert det_fs[0]["class"] == det_stock[0]["class"] == "FEMALE_BREAST_EXPOSED"
    assert det_fs[0]["box"] == det_stock[0]["box"]
    assert abs(det_fs[0]["score"] - det_stock[0]["score"]) < 1e-5
