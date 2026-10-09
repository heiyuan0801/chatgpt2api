from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from services.config import config
from utils.diagnostics import diagnostic_excerpt
from utils.image_tokens import image_size_from_bytes
from utils.log import logger


_FINAL2X_MODEL = os.getenv(
    "CHATGPT2API_FINAL2X_MODEL",
    "realesr-general-x4v3.pth",
)
_FINAL2X_DEVICE = os.getenv("CHATGPT2API_FINAL2X_DEVICE", "cpu")


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


_FINAL2X_TIMEOUT_SECONDS = _env_int("CHATGPT2API_FINAL2X_TIMEOUT_SECONDS", 600, 30)
_SHARP_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "image_upscale" / "upscale.mjs"
_UPSCALE_LOCK = threading.Lock()
_FINAL2X_WRAPPER: object | None = None
_FINAL2X_WRAPPER_KEY: tuple[str, str, int] | None = None
_MAX_TARGET_DIMENSION = 8192
_GENERAL_FAST_MODEL = "realesr-general-x4v3.pth"
_GENERAL_FAST_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-x4v3.pth"
)


class ImageUpscaleError(RuntimeError):
    """Raised when an explicitly requested AI upscale cannot be delivered."""

    code = "image_upscale_failed"


@dataclass(frozen=True)
class ImageUpscaleResult:
    data: bytes
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    engine: str


def _target_size(value: object, source: tuple[int, int] | None = None) -> tuple[int, int] | None:
    text = str(value or "").strip().lower()
    if text in {"", "auto", "original", "1k"}:
        return None
    if text in {"2k", "2048"}:
        longest = 2048
    elif text in {"4k", "3840"}:
        longest = 3840
    else:
        match = re.fullmatch(r"(\d{2,5})\s*x\s*(\d{2,5})", text)
        if not match:
            return None
        width, height = int(match.group(1)), int(match.group(2))
        if width > _MAX_TARGET_DIMENSION or height > _MAX_TARGET_DIMENSION:
            return None
        if not source or source[0] <= 0 or source[1] <= 0:
            return width, height
        ratio = min(width / source[0], height / source[1])
        return max(1, round(source[0] * ratio)), max(1, round(source[1] * ratio))
    if not source:
        return longest, longest
    width, height = source
    if width <= 0 or height <= 0:
        return None
    ratio = longest / max(width, height)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def _pillow_lanczos(image_data: bytes, target: tuple[int, int]) -> bytes:
    with Image.open(io.BytesIO(image_data)) as source:
        image_format = str(source.format or "PNG").upper()
        image = ImageOps.exif_transpose(source)
        resized = image.resize(target, Image.Resampling.LANCZOS, reducing_gap=3.0)
        output = io.BytesIO()
        save_options: dict[str, object] = {}
        if image_format in {"JPG", "JPEG"}:
            image_format = "JPEG"
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            save_options.update(quality=95, subsampling=0)
        elif image_format == "WEBP":
            save_options.update(quality=95, method=4)
        elif image_format not in {"PNG", "WEBP"}:
            image_format = "PNG"
        resized.save(output, format=image_format, **save_options)
        return output.getvalue()


def _final2x(image_data: bytes, target: tuple[int, int]) -> bytes:
    source = image_size_from_bytes(image_data)
    if not source:
        raise ImageUpscaleError("无法读取原图尺寸")
    model_scale = 2 if "_2x" in _FINAL2X_MODEL.lower() else 4

    # Keep the model resident in this worker.  Starting Final2x-core for every
    # image reloads the model and adds several seconds before inference starts.
    # Imports stay lazy so the original-image path does not pay the AI runtime
    # startup cost.
    try:
        import cv2
        import numpy as np
        from Final2x_core.SRclass import SRWrapper
        from Final2x_core.config import SRConfig
    except ImportError as exc:
        raise ImageUpscaleError("Final2x-core 运行时不可用") from exc

    global _FINAL2X_WRAPPER, _FINAL2X_WRAPPER_KEY
    wrapper_key = (_FINAL2X_MODEL, _FINAL2X_DEVICE, model_scale)
    if _FINAL2X_WRAPPER_KEY != wrapper_key:
        try:
            if _FINAL2X_MODEL == _GENERAL_FAST_MODEL:
                from cccv.config import CONFIG_REGISTRY
                from cccv.config.sr.realesrgan_config import RealESRGANConfig
                from cccv.type import ArchType, ModelType

                if _GENERAL_FAST_MODEL not in CONFIG_REGISTRY:
                    CONFIG_REGISTRY.register(
                        RealESRGANConfig(
                            name=_GENERAL_FAST_MODEL,
                            url=_GENERAL_FAST_MODEL_URL,
                            arch=ArchType.SRVGG,
                            model=ModelType.SRBaseModel,
                            scale=4,
                            num_conv=32,
                        )
                    )
            wrapper = SRWrapper(
                SRConfig(
                    pretrained_model_name=_FINAL2X_MODEL,
                    device=_FINAL2X_DEVICE,
                    use_tile=True,
                    precision="fp32",
                    target_scale=model_scale,
                    output_path=Path(tempfile.gettempdir()),
                    input_path=[Path(__file__)],
                    save_format=".png",
                )
            )
        except Exception as exc:
            raise ImageUpscaleError(f"Final2x-core 模型初始化失败: {diagnostic_excerpt(str(exc), 500)}") from exc
        _FINAL2X_WRAPPER = wrapper
        _FINAL2X_WRAPPER_KEY = wrapper_key

    encoded = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageUpscaleError("Final2x-core 无法读取原图")
    alpha = None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]
    try:
        result = _FINAL2X_WRAPPER.process(image)  # type: ignore[union-attr]
        if alpha is not None:
            alpha_rgb = np.repeat(alpha[:, :, None], 3, axis=2)
            alpha_result = _FINAL2X_WRAPPER.process(alpha_rgb)  # type: ignore[union-attr]
            result = np.dstack((result, alpha_result[:, :, 0]))
        ok, output = cv2.imencode(".png", result)
    except Exception as exc:
        raise ImageUpscaleError(f"Final2x-core 推理失败: {diagnostic_excerpt(str(exc), 500)}") from exc
    if not ok:
        raise ImageUpscaleError("Final2x-core 输出图片编码失败")
    return output.tobytes()


def upscale_image(
    image_data: bytes,
    requested_size: object = None,
    *,
    enabled: bool | None = None,
    target: object = None,
) -> ImageUpscaleResult:
    """Run Final2x for an explicit request and converge to the exact target size."""
    source = image_size_from_bytes(image_data)
    if not source:
        raise ImageUpscaleError("无法读取原图尺寸")
    should_upscale = config.image_upscale_enabled if enabled is None else bool(enabled)
    target_value = target if target is not None else requested_size
    if should_upscale and target_value in (None, ""):
        target_value = "4k"
    target_size = _target_size(target_value, source)
    if not should_upscale or not target_size:
        return ImageUpscaleResult(image_data, source, source, "original")
    if max(source) >= max(target_size):
        return ImageUpscaleResult(image_data, source, source, "original")

    engine = config.image_upscale_engine
    with _UPSCALE_LOCK:
        if engine == "final2x":
            result = _final2x(image_data, target_size)
        elif engine == "sharp_lanczos3":
            node = shutil.which("node")
            if not node or not _SHARP_SCRIPT.is_file():
                raise ImageUpscaleError("Sharp runtime 不可用")
            completed = subprocess.run(
                [node, str(_SHARP_SCRIPT), str(target_size[0]), str(target_size[1])],
                input=image_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=_FINAL2X_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0 or not completed.stdout:
                raise ImageUpscaleError(completed.stderr.decode("utf-8", errors="replace") or "Sharp 执行失败")
            result = completed.stdout
        else:
            result = _pillow_lanczos(image_data, target_size)

        result_size = image_size_from_bytes(result)
        if not result_size:
            raise ImageUpscaleError("超分输出图片无效")
        if result_size != target_size:
            result = _pillow_lanczos(result, target_size)
            result_size = image_size_from_bytes(result) or target_size
        logger.info({
            "event": "image_upscale_done",
            "engine": engine,
            "source_size": list(source),
            "target_size": list(target_size),
            "result_size": list(result_size),
            "source_bytes": len(image_data),
            "result_bytes": len(result),
        })
        return ImageUpscaleResult(result, source, result_size, engine)


def upscale_image_if_needed(
    image_data: bytes,
    requested_size: object = None,
    *,
    enabled: bool | None = None,
    target: object = None,
) -> bytes:
    """Backward-compatible byte-only adapter for existing callers."""
    return upscale_image(
        image_data,
        requested_size,
        enabled=enabled,
        target=target,
    ).data
