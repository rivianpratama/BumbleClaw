from __future__ import annotations

from pathlib import Path

import gradio as gr

from face_similarity.embedding import get_face_embedding
from face_similarity.scoring import score_embedding
from face_similarity.store import load_store

STORE_PATH = "embeddings/reference_store.npz"
TOP_K = 11


def analyze(image_path: str) -> str:
    if not image_path:
        return "Upload an image to score."

    try:
        store = load_store(STORE_PATH)
        embedding = get_face_embedding(
            image_path,
            model_name=store.model_name,
            provider=store.provider,
            det_size=store.det_size,
            det_thresh=store.det_thresh,
            enforce_detection=True,
        )
        result = score_embedding(embedding, store.embeddings, store.ratings, k=TOP_K)
    except Exception as exc:
        return f"Error: {exc}"

    nearest_lines = []
    for index, similarity in zip(result.nearest_indices, result.nearest_similarities):
        nearest_lines.append(
            f"- {Path(store.paths[index]).name}: rating {store.ratings[index]:.0f}, similarity {similarity:.4f}"
        )

    nearest_text = "\n".join(nearest_lines)
    return (
        f"## Rating: {result.rating:.1f}/100\n\n"
        f"- Max similarity: `{result.max_similarity:.4f}`\n"
        f"- Mean similarity: `{result.mean_similarity:.4f}`\n"
        f"- Compared against: `{len(store.embeddings)}` reference images\n\n"
        f"### Nearest references\n{nearest_text}"
    )


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="filepath", label="Upload face image"),
    outputs=gr.Markdown(label="Result"),
    title="Face Similarity Rating",
    description="Scores a face from 0-100 by comparing it to your labeled reference examples.",
)


if __name__ == "__main__":
    demo.launch(share=False)
