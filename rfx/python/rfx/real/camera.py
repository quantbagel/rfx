"""
rfx.real.camera - Camera interfaces
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class Camera:
    """USB camera interface using OpenCV."""

    def __init__(self, device_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self._cap = None

    def _init_camera(self):
        try:
            import cv2

            self._cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")

    def capture(self) -> torch.Tensor:
        if self._cap is None:
            self._init_camera()
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.release()


class RealSenseCamera:
    """Intel RealSense RGB-D camera interface."""

    def __init__(
        self,
        serial_number: Optional[str] = None,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        align_depth: bool = True,
    ):
        self.serial_number = serial_number
        self.resolution = resolution
        self.fps = fps
        self.align_depth = align_depth
        self._pipeline = None
        self._align = None

    def _init_camera(self):
        try:
            import pyrealsense2 as rs
            import numpy as np

            self._rs = rs
            self._np = np
        except ImportError:
            raise ImportError("pyrealsense2 not installed. Install with: pip install pyrealsense2")

        self._pipeline = rs.pipeline()
        config = rs.config()

        if self.serial_number:
            config.enable_device(self.serial_number)

        config.enable_stream(
            rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps
        )
        config.enable_stream(
            rs.stream.color, self.resolution[0], self.resolution[1], rs.format.rgb8, self.fps
        )

        self._pipeline.start(config)

        if self.align_depth:
            self._align = rs.align(rs.stream.color)

    def capture(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._pipeline is None:
            self._init_camera()

        frames = self._pipeline.wait_for_frames()
        if self._align:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture frames")

        depth_image = self._np.asanyarray(depth_frame.get_data())
        color_image = self._np.asanyarray(color_frame.get_data())

        depth_tensor = torch.from_numpy(depth_image.astype(self._np.float32)) / 1000.0
        color_tensor = torch.from_numpy(color_image)

        return color_tensor, depth_tensor

    def capture_rgb(self) -> torch.Tensor:
        rgb, _ = self.capture()
        return rgb

    def release(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def __del__(self):
        self.release()
