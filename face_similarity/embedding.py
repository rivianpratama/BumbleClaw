from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import site
import tempfile
from typing import Any

import numpy as np

DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_PROVIDER = "auto"
DEFAULT_DET_SIZE = 640
DEFAULT_DET_THRESH = 0.5
_DLL_DIRECTORY_HANDLES: list[Any] = []


def get_face_embedding(
    image_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    provider: str = DEFAULT_PROVIDER,
    det_size: int = DEFAULT_DET_SIZE,
    det_thresh: float = DEFAULT_DET_THRESH,
    enforce_detection: bool = True,
) -> np.ndarray:
    """Extract one InsightFace embedding and prefer the largest detected face."""
    cv2 = _import_cv2()
    app, _ = get_face_app(model_name=model_name, provider=provider, det_size=det_size, det_thresh=det_thresh)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(image)
    if not faces:
        if enforce_detection:
            raise ValueError(f"No face detected in image: {image_path}")
        raise ValueError(f"No face detected in image, even with lenient mode: {image_path}")

    face = select_largest_face(faces)
    return np.asarray(face.embedding, dtype=np.float32)


@lru_cache(maxsize=8)
def get_face_app(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    provider: str = DEFAULT_PROVIDER,
    det_size: int = DEFAULT_DET_SIZE,
    det_thresh: float = DEFAULT_DET_THRESH,
) -> tuple[Any, str]:
    insightface = _import_insightface()
    providers, active_provider = resolve_providers(provider)
    app = insightface.app.FaceAnalysis(name=model_name, providers=providers)
    ctx_id = 0 if active_provider == "cuda" else -1
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
    actual_provider = active_provider_from_app(app)
    if provider.lower() == "cuda" and actual_provider != "cuda":
        raise RuntimeError(
            "CUDAExecutionProvider is listed by ONNX Runtime, but model sessions fell back to CPU. "
            "Install the missing NVIDIA CUDA/cuDNN runtime DLLs or use --provider cpu."
        )
    return app, actual_provider


def resolve_providers(provider: str = DEFAULT_PROVIDER) -> tuple[list[str], str]:
    requested = provider.lower()
    available = available_onnx_providers()
    has_cuda = "CUDAExecutionProvider" in available
    cuda_works = has_cuda and cuda_provider_works()

    if requested == "auto":
        if cuda_works:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
        return ["CPUExecutionProvider"], "cpu"
    if requested == "cuda":
        if not has_cuda:
            available_text = ", ".join(available) or "none"
            raise RuntimeError(
                "CUDAExecutionProvider is not available to ONNX Runtime. "
                f"Available providers: {available_text}"
            )
        if not cuda_works:
            raise RuntimeError(
                "CUDAExecutionProvider is listed by ONNX Runtime, but it is not usable. "
                "Install CUDA 12.x, cuDNN 9.x, and the latest MSVC runtime, then ensure their DLLs are on PATH."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
    if requested == "cpu":
        return ["CPUExecutionProvider"], "cpu"
    raise ValueError("provider must be one of: auto, cuda, cpu")


def available_onnx_providers() -> list[str]:
    onnxruntime = _import_onnxruntime()
    return list(onnxruntime.get_available_providers())


@lru_cache(maxsize=1)
def cuda_provider_works() -> bool:
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError:
        return False

    input_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "cuda_check", [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cuda_check.onnx"
        onnx.save(model, path)
        session = ort.InferenceSession(
            str(path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in session.get_providers():
            return False
        result = session.run(None, {"x": np.asarray([1.0], dtype=np.float32)})
        return float(result[0][0]) == 1.0


def active_provider_from_app(app: Any) -> str:
    for model in getattr(app, "models", {}).values():
        session = getattr(model, "session", None)
        if session is None:
            continue
        providers = session.get_providers()
        if "CUDAExecutionProvider" in providers:
            return "cuda"
    return "cpu"


def select_largest_face(faces: list[Any]) -> Any:
    if not faces:
        raise ValueError("No face detected")

    def area(face: Any) -> float:
        x1, y1, x2, y2 = face.bbox
        return float(max(x2 - x1, 0) * max(y2 - y1, 0))

    return max(faces, key=area)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required. Run: pip install -r requirements.txt") from exc
    return cv2


def _import_insightface() -> Any:
    try:
        import insightface
    except ImportError as exc:
        raise ImportError("insightface is required. Run: pip install -r requirements.txt") from exc
    return insightface


def _import_onnxruntime() -> Any:
    add_nvidia_dll_directories()
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError("onnxruntime-gpu is required. Run: pip install -r requirements.txt") from exc
    return onnxruntime


@lru_cache(maxsize=1)
def add_nvidia_dll_directories() -> None:
    if os.name != "nt":
        return

    candidates: list[Path] = []
    for package_root in site.getsitepackages():
        nvidia_root = Path(package_root) / "nvidia"
        candidates.extend(nvidia_root.glob("*\\bin"))

    existing_path = os.environ.get("PATH", "")
    existing_parts = existing_path.split(os.pathsep) if existing_path else []
    new_parts = []
    for path in candidates:
        if path.exists():
            path_text = str(path)
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(path_text))
            if path_text not in existing_parts:
                new_parts.append(path_text)

    if new_parts:
        os.environ["PATH"] = os.pathsep.join([*new_parts, existing_path])
