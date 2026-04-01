"""Stream camera frames to the rfx platform via WebSocket.

Runs in a background thread so the main control loop is not blocked.
Uses only stdlib + opencv for zero extra dependencies beyond what
``rfx[teleop]`` already provides.
"""

from __future__ import annotations

import io
import logging
import struct
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class CameraStreamer:
    """Capture from local cameras and push JPEG frames to the platform.

    Args:
        platform_url: Base URL of the platform (http or https).
        robot_id: Robot identifier (must match the registered robot).
        camera_ids: OpenCV device indices to capture from.
        camera_names: Human-readable names for each camera.
        fps: Target capture rate.
        jpeg_quality: JPEG compression quality (1-100).
    """

    def __init__(
        self,
        platform_url: str,
        robot_id: str,
        camera_ids: list[int],
        camera_names: list[str] | None = None,
        fps: float = 10,
        jpeg_quality: int = 70,
    ) -> None:
        self._platform_url = platform_url.rstrip("/")
        self._robot_id = robot_id
        self._camera_ids = camera_ids
        self._camera_names = camera_names or [f"camera_{i}" for i in camera_ids]
        self._fps = fps
        self._jpeg_quality = jpeg_quality

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def ws_url(self) -> str:
        base = self._platform_url
        # Convert http(s) to ws(s)
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        elif not base.startswith("ws"):
            base = "ws://" + base

        # The dashboard proxies REST but not WebSocket, so if the URL
        # looks like a dashboard URL (has /api prefix in platform_client calls),
        # we connect to the backend camera endpoint directly.
        names = ",".join(self._camera_names)
        return f"{base}/cameras/{self._robot_id}/feed?cameras={names}"

    def start(self) -> None:
        """Start streaming in a background daemon thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="rfx-camera-streamer",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "Camera streamer started: %d camera(s) at %d fps → %s",
            len(self._camera_ids),
            self._fps,
            self.ws_url,
        )

    def stop(self) -> None:
        """Signal stop and wait for the thread to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        try:
            import cv2
        except ImportError:
            log.error("opencv-python not installed. Camera streaming requires: pip install opencv-python")
            return

        try:
            import websockets.sync.client as ws_sync
        except ImportError:
            log.error("websockets not installed. Camera streaming requires: pip install websockets")
            return

        # Open cameras
        caps = []
        for device_id in self._camera_ids:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                log.error("Failed to open camera %d", device_id)
                return
            caps.append(cap)

        interval = 1.0 / self._fps
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]

        try:
            ws = ws_sync.connect(self.ws_url)
        except Exception as exc:
            log.error("WebSocket connect failed: %s", exc)
            for cap in caps:
                cap.release()
            return

        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                for cam_idx, cap in enumerate(caps):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    ok, buf = cv2.imencode(".jpg", frame, encode_params)
                    if not ok:
                        continue
                    # Wire format: [1 byte camera_index][JPEG bytes]
                    ws.send(bytes([cam_idx]) + buf.tobytes())

                elapsed = time.monotonic() - t0
                sleep_s = interval - elapsed
                if sleep_s > 0:
                    self._stop.wait(sleep_s)
        except Exception as exc:
            if not self._stop.is_set():
                log.warning("Camera streamer error: %s", exc)
        finally:
            try:
                ws.close()
            except Exception:
                pass
            for cap in caps:
                cap.release()
            log.info("Camera streamer stopped")
