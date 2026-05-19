from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from face_similarity.clip_embedding import import_clip_dependencies, select_device
from face_similarity.clip_store import DEFAULT_CLIP_MODEL, save_clip_store
from face_similarity.labels import load_labels
from face_similarity.store import load_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP image embeddings aligned to a face reference store.")
    parser.add_argument("--labels", default="dataset_labels.csv", help="CSV with labeled image paths")
    parser.add_argument("--store", default="embeddings/reference_store.npz", help="Face reference store to align with")
    parser.add_argument("--output", default="embeddings/clip_store.npz", help="Output CLIP store path")
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL, help="Hugging Face CLIP model name")
    parser.add_argument("--provider", default="auto", choices=["auto", "cuda", "cpu"], help="Torch device preference")
    parser.add_argument("--batch-size", type=int, default=32, help="Image batch size")
    parser.add_argument("--limit", type=int, help="Optional maximum number of aligned images for smoke tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_labels(args.labels)  # validates the CSV without controlling store alignment
    face_store = load_store(args.store)
    paths = [Path(path) for path in face_store.paths]
    if args.limit is not None:
        paths = paths[: args.limit]

    torch, CLIPModel, CLIPProcessor = import_clip_dependencies()
    device = select_device(torch, args.provider)
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    kept_paths: list[str] = []
    progress = tqdm(range(0, len(paths), args.batch_size), desc="Embedding CLIP images")
    for start in progress:
        batch_paths = paths[start : start + args.batch_size]
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_paths.append(path)
            except Exception as exc:
                print(f"SKIP {path}: {exc}")
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        batch_embeddings = features.detach().cpu().numpy().astype(np.float32)
        embeddings.extend(batch_embeddings)
        kept_paths.extend(str(path) for path in valid_paths)

    save_clip_store(
        args.output,
        embeddings=embeddings,
        paths=kept_paths,
        model_name=args.model,
        device=device,
    )
    print(f"Saved {len(embeddings)} CLIP embedding(s) to {args.output}")
    print(f"Device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
