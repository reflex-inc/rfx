"""Stream camera frames to the rfx platform via WebSocket.

Runs in a background thread so the main control loop is not blocked.
Supports Intel RealSense (D405, etc.) and USB cameras via OpenCV.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


def _detect_realsense_cameras() -> list[str]:
    """Return serial numbers of connected RealSense devices."""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
    except ImportError:
        return []
    except Exception:
        return []


class CameraStreamer:
    """Capture from cameras and push JPEG frames to the platform.

    Supports two backends:
    - ``realsense``: Intel RealSense cameras (D405, D435, etc.)
    - ``cv2``: USB cameras via OpenCV

    The backend is auto-detected: if pyrealsense2 is installed and
    RealSense devices are found, it uses RealSense. Otherwise OpenCV.

    Args:
        platform_url: Base URL of the platform (http or https).
        robot_id: Robot identifier (must match the registered robot).
        camera_ids: Device indices (OpenCV) or serial numbers (RealSense).
            For RealSense, pass empty list to auto-detect all connected devices.
        camera_names: Human-readable names for each camera.
        fps: Target capture rate.
        jpeg_quality: JPEG compression quality (1-100).
        backend: ``"auto"``, ``"realsense"``, or ``"cv2"``.
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
        self._camera_names = camera_names
        self._fps = fps
        self._jpeg_quality = jpeg_quality
        self._backend = backend

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _resolve_backend(self) -> str:
        if self._backend != "auto":
            return self._backend
        serials = _detect_realsense_cameras()
        if serials:
            return "realsense"
        return "cv2"

    @property
    def ws_url(self) -> str:
        base = self._platform_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        elif not base.startswith("ws"):
            base = "ws://" + base
        names = ",".join(self._camera_names or [])
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

    def _run(self) -> None:
        try:
            import websockets.sync.client as ws_sync
        except ImportError:
            log.error("websockets not installed: pip install websockets")
            return

        backend = self._resolve_backend()

        if backend == "realsense":
            self._run_realsense(ws_sync)
        else:
            self._run_cv2(ws_sync)

    def _run_realsense(self, ws_sync) -> None:
        import pyrealsense2 as rs
        import numpy as np

        try:
            import cv2
        except ImportError:
            log.error("opencv-python needed for JPEG encoding: pip install opencv-python")
            return

        ctx = rs.context()
        devices = ctx.query_devices()
        if not devices:
            log.error("No RealSense devices found")
            return

        # Resolve which devices to use
        if self._camera_ids:
            serials = [str(s) for s in self._camera_ids]
        else:
            serials = [d.get_info(rs.camera_info.serial_number) for d in devices]

        if not self._camera_names:
            self._camera_names = [f"realsense_{i}" for i in range(len(serials))]

        log.info(
            "RealSense streamer: %d camera(s) at %d fps",
            len(serials), self._fps,
        )

        # Start pipelines
        pipelines = []
        for serial in serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, int(self._fps))
            try:
                pipeline.start(config)
                pipelines.append(pipeline)
                log.info("  RealSense %s started", serial)
            except Exception as exc:
                log.error("  RealSense %s failed: %s", serial, exc)

        if not pipelines:
            log.error("No RealSense pipelines started")
            return

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        interval = 1.0 / self._fps

        try:
            ws = ws_sync.connect(self.ws_url)
        except Exception as exc:
            log.error("WebSocket connect failed: %s", exc)
            for p in pipelines:
                p.stop()
            return

        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                for cam_idx, pipeline in enumerate(pipelines):
                    try:
                        frames = pipeline.wait_for_frames(timeout_ms=100)
                        color_frame = frames.get_color_frame()
                        if not color_frame:
                            continue
                        img = np.asanyarray(color_frame.get_data())
                        ok, buf = cv2.imencode(".jpg", img, encode_params)
                        if ok:
                            ws.send(bytes([cam_idx]) + buf.tobytes())
                    except Exception:
                        continue

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
            for p in pipelines:
                try:
                    p.stop()
                except Exception:
                    pass
            log.info("Camera streamer stopped")

    def _run_cv2(self, ws_sync) -> None:
        try:
            import cv2
        except ImportError:
            log.error("opencv-python not installed: pip install opencv-python")
            return

        camera_ids = [int(c) for c in self._camera_ids] if self._camera_ids else [0]

        if not self._camera_names:
            self._camera_names = [f"camera_{i}" for i in camera_ids]

        log.info(
            "OpenCV streamer: %d camera(s) at %d fps",
            len(camera_ids), self._fps,
        )

        caps = []
        for device_id in camera_ids:
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
