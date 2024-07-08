#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from corenet.data.transforms.common import BaseTransformation, Identity
from corenet.data.video_reader.ffmpeg_reader import FFMPEGReader
from corenet.data.video_reader.ffmpeg_utils import (
    IS_FFMPEG_INSTALLED,
    get_flags_to_replicate_audio_codec,
    transform_video_file,
)
from tests.configs import get_config


def assert_video_data_is_close(actual: Dict, expected: Dict) -> None:
    if not IS_FFMPEG_INSTALLED:
        pytest.skip("FFmpeg (optional dependncy) is not installed.")
    """Given two ffmpeg reader outputs, asserts audio and video signals are close."""

    def matching_ratio(x: torch.Tensor, y: torch.Tensor, atol=0.05, rtol=0.0):
        """Returns the percentage of elements of x,y that are close with atol, rtol."""
        return torch.isclose(x, y, atol=atol, rtol=rtol).float().mean()

    assert matching_ratio(actual["video"], expected["video"]) >= 0.99

    assert np.isclose(
        actual["audio"].shape[0],
        expected["audio"].shape[0],
        atol=0.05 * actual["metadata"]["audio_fps"],  # 0.05 seconds
    )
    min_length = min(actual["audio"].shape[0], expected["audio"].shape[0])
    assert (
        matching_ratio(actual["audio"][:min_length], expected["audio"][:min_length])
        >= 0.99
    )


def test_transform_video_with_trimming(tmp_path: Path):
    if not IS_FFMPEG_INSTALLED:
        pytest.skip("FFmpeg (optional dependncy) is not installed.")
    opts = get_config()
    input_path = "./tests/data/dummy_video.mov"
    output_path = str(tmp_path / "output.mov")
    from_timestamp = 1
    to_timestamp = 2.5
    transform_video_file(
        input_filename=input_path,
        output_filename=output_path,
        transform=Identity(opts),
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        encoder_flags={
            # Audio codec should be specified when trimming.
            **get_flags_to_replicate_audio_codec(input_path),
            # Use highest possible quality so that we can compare the results.
            "q:a": 0,
            "q:v": 0,
        },
    )

    reader = FFMPEGReader(opts)
    input_data = reader.read_video(
        input_path,
    )
    output_data = reader.read_video(
        output_path,
    )

    expected_data = copy.deepcopy(input_data)
    for key in ("video", "audio"):
        fps = expected_data["metadata"][f"{key}_fps"]
        from_frame = int(from_timestamp * fps)
        to_frame = int(to_timestamp * fps)
        expected_data[key] = expected_data[key][from_frame:to_frame]

    assert_video_data_is_close(actual=output_data, expected=expected_data)


class SmallCrop(BaseTransformation):
    CROP_SIZE = 7, 9

    def __call__(self, data: Dict) -> Dict:
        frames = data["samples"]["video"]
        frames = frames[:, :, :, : self.CROP_SIZE[0], : self.CROP_SIZE[1]]
        data["samples"]["video"] = frames
        return data


@pytest.mark.parametrize(
    "transform_cls,transform_kwargs",
    [(Identity, {}), (SmallCrop, {"output_dimensions": SmallCrop.CROP_SIZE})],
)
def test_transform_video_file(
    tmp_path: Path,
    transform_cls: BaseTransformation,
    transform_kwargs: Dict,
):
    if not IS_FFMPEG_INSTALLED:
        pytest.skip("FFmpeg (optional dependncy) is not installed.")
    opts = get_config()
    input_path = "./tests/data/dummy_video.mov"
    output_path = str(tmp_path / "output.mov")
    transform_video_file(
        input_filename=input_path,
        output_filename=output_path,
        transform=transform_cls(opts),
        encoder_flags={
            # Use highest possible quality so that we can compare the results.
            "q:a": 0,
            "q:v": 0,
        },
        **transform_kwargs,
    )

    reader = FFMPEGReader(opts)
    input_data = reader.read_video(input_path)
    output_data = reader.read_video(output_path)

    for data in input_data, output_data:
        # The BaseTransformation instances expect 5D tensors for multi-clip videos.
        # FIXME: A better solution is to refactor FFMPEGReader.read_video() to follow
        # the convention of returning a 5D tensor, rather than a 4D tensor.
        data["video"] = torch.unsqueeze(data["video"], 0)

    # Apply the transformation on all frames of the input data. `output_data` is
    # transformed chunk-by-chunk using an ffmpeg decode->transform->concat->encode
    # pipeline, while `input_data` is read into RAM as a whole and processed using the
    # same transformation.
    expected_data = transform_cls(opts)(
        {
            "samples": input_data,
            "targets": {},
        }
    )["samples"]

    assert_video_data_is_close(actual=output_data, expected=expected_data)
