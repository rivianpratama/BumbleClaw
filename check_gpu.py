from __future__ import annotations

from face_similarity.embedding import available_onnx_providers, cuda_provider_works


def main() -> int:
    providers = available_onnx_providers()
    print("ONNX Runtime providers:")
    for provider in providers:
        print(f"- {provider}")

    if "CUDAExecutionProvider" in providers and cuda_provider_works():
        print("\nCUDA GPU is available and usable.")
        return 0

    print("\nCUDA GPU is not usable. The app will use CPU unless this is fixed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
