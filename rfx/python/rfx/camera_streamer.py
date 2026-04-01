"""Stream camera frames to the rfx platform via WebSocket.

Runs in a background thread so the main control loop is not blocked.
Supports Intel RealSense (D405, etc.), USB cameras via OpenCV, or both mixed.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class _RealSenseSource:
    """Wraps a RealSense pipeline as a frame source."""

    def __init__(self, serial: str, fps: int):
        import pyrealsense2 as rs

        self.serial = serial
        self._pipeline = None
        self._rs = rs

        _CONFIGS = [
            (rs.stream.color, 640, 480, rs.format.bgr8),
            (rs.stream.color, 424, 240, rs.format.bgr8),
            (rs.stream.color, 848, 480, rs.format.bgr8),
            (rs.stream.color, 1280, 720, rs.format.bgr8),
        ]

        # Try explicit configs first
        for stream_type, w, h, fmt in _CONFIGS:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(stream_type, w, h, fmt, fps)
            try:
                pipeline.start(config)
                self._pipeline = pipeline
                log.info("  %s started (%dx%d)", serial, w, h)
                return
            except Exception:
                continue

        # Fallback: let SDK pick
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        try:
            pipeline.start(config)
            self._pipeline = pipeline
            log.info("  %s started (default config)", serial)
        except Exception as exc:
            raise RuntimeError(f"RealSense {serial}: {exc}") from exc

    def capture(self):
        """Returns a BGR numpy array or None."""
        import numpy as np
        try:
            import cv2
        except ImportError:
            return None

        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=100)
        except Exception:
            return None

        color = frames.get_color_frame()
        if color:
            return np.asanyarray(color.get_data())

        ir = frames.get_infrared_frame()
        if ir:
            arr = np.asanyarray(ir.get_data())
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else arr
        return None

    def release(self):
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


class _CV2Source:
    """Wraps an OpenCV VideoCapture as a frame source."""

    def __init__(self, device_id: int):
        import cv2
        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device_id}")
        log.info("  camera_%d started", device_id)

    def capture(self):
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None


class CameraStreamer:
    """Capture from any mix of cameras and push JPEG frames to the platform.

    Args:
        platform_url: Base URL of the platform.
        robot_id: Robot identifier.
        camera_ids: List of device IDs. RealSense serials (strings like "352122271378")
            and OpenCV indices (strings like "0", "1") can be mixed.
        camera_names: Human-readable names for each camera.
        fps: Target capture rate.
        jpeg_quality: JPEG compression quality (1-100).
        backend: "auto", "realsense", "cv2", or "mixed".
    """

    def __init__(
        self,
        platform_url: str,
        robot_id: str,
        camera_ids: list[int | str] | None = None,
        camera_names: list[str] | None = None,
        fps: float = 10,
        jpeg_quality: int = 70,
        backend: str = "auto",
    ) -> None:
        self._platform_url = platform_url.rstrip("/")
        self._robot_id = robot_id
        self._camera_ids = camera_ids or []
        self._camera_names = camera_names or []
        self._fps = fps
        self._jpeg_quality = jpeg_quality
        self._backend = backend

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def ws_url(self) -> str:
        base = self._platform_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        elif not base.startswith("ws"):
            base = "ws://" + base
        names = ",".join(self._camera_names)
        return f"{base}/cameras/{self._robot_id}/feed?cameras={names}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="rfx-camera-streamer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _is_realsense_id(self, cam_id: str) -> bool:
        """RealSense serials are long numeric strings (10+ digits)."""
        return len(cam_id) >= 6 and cam_id.isdigit() and int(cam_id) > 100

    def _build_sources(self) -> list[_RealSenseSource | _CV2Source]:
        """Create a frame source for each camera ID."""
        rs_available = False
        try:
            import pyrealsense2  # noqa: F401
            rs_available = True
        except ImportError:
            pass

        sources = []
        for cam_id in self._camera_ids:
            sid = str(cam_id)
            if rs_available and self._is_realsense_id(sid):
                try:
                    sources.append(_RealSenseSource(sid, int(self._fps)))
                except Exception as exc:
                    log.error("  %s", exc)
            else:
                try:
                    sources.append(_CV2Source(int(sid)))
                except Exception as exc:
                    log.error("  %s", exc)
        return sources

    def _run(self) -> None:
        try:
            import websockets.sync.client as ws_sync
        except ImportError:
            log.error("websockets not installed: pip install websockets")
            return

        try:
            import cv2
        except ImportError:
            log.error("opencv-python not installed: pip install opencv-python")
            return

        sources = self._build_sources()
        if not sources:
            log.error("No cameras started")
            return

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        interval = 1.0 / self._fps

        try:
            ws = ws_sync.connect(self.ws_url)
        except Exception as exc:
            log.error("WebSocket connect failed: %s", exc)
            for s in sources:
                s.release()
            return

        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                for cam_idx, source in enumerate(sources):
                    img = source.capture()
                    if img is None:
                        continue
                    ok, buf = cv2.imencode(".jpg", img, encode_params)
                    if ok:
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
            for s in sources:
                s.release()
            log.info("Camera streamer stopped")
