"""fps→frame step and timestamp format."""

from framesentry.core.timecode import format_timestamp, frame_step_from_fps


def test_frame_step_basic():
    assert frame_step_from_fps(30.0, 2.0) == 15
    assert frame_step_from_fps(25.0, 1.0) == 25
    assert frame_step_from_fps(24.0, 8.0) == 3


def test_frame_step_sample_ge_video():
    assert frame_step_from_fps(10.0, 10.0) == 1
    assert frame_step_from_fps(10.0, 30.0) == 1


def test_frame_step_invalid():
    assert frame_step_from_fps(0, 2) == 1
    assert frame_step_from_fps(30, 0) == 1


def test_format_timestamp():
    assert format_timestamp(0) == "00-00-00.000"
    assert format_timestamp(462.251) == "00-07-42.251"
    assert format_timestamp(3661.5) == "01-01-01.500"


def test_format_timestamp_negative_safe():
    assert format_timestamp(-1) == "00-00-00.000"
