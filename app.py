from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import gradio as gr

from face_similarity.clip_embedding import get_clip_embedding
from face_similarity.embedding import get_face_embedding
from face_similarity.regressor import (
    load_regressor,
    predict_multimodal_rating,
    predict_rating,
)
from face_similarity.scoring import score_embedding
from face_similarity.store import load_store

STORE_PATH = "embeddings/reference_store_bumble_combined_round2.npz"
REGRESSOR_PATH = "models/rating_regressor_bumble_combined_round2.joblib"
MULTIMODAL_REGRESSOR_PATH = "models/rating_regressor_multimodal_bumble_combined_round2.joblib"
ORIGINAL_REGRESSOR_PATH = "models/rating_regressor.joblib"
ORIGINAL_MULTIMODAL_REGRESSOR_PATH = "models/rating_regressor_multimodal.joblib"
ROUND1_REGRESSOR_PATH = "models/rating_regressor_bumble_combined.joblib"
ROUND1_MULTIMODAL_REGRESSOR_PATH = "models/rating_regressor_multimodal_bumble_combined.joblib"
TOP_K = 20
BIASED_FACE_WEIGHT = 0.22
COMPARISON_FACE_WEIGHT = 0.50
SWIPE_THRESHOLD = 54.0
SERVER_PORT = int(os.environ.get("BUMBLECLAW_PORT", "7860"))

CSS = """
body {
    background: #f7f7f4;
}

.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto;
}

#app-shell {
    min-height: 100vh;
}

#title {
    margin: 10px 0 8px;
}

#title h1 {
    font-size: 24px;
    line-height: 1.15;
    margin: 0;
    letter-spacing: 0;
}

#image-input {
    border-radius: 8px;
    overflow: hidden;
}

#score-panel {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 10px;
}

.score-card {
    background: #ffffff;
    border: 1px solid #deded8;
    border-radius: 8px;
    padding: 12px;
}

.score-label {
    color: #64645d;
    font-size: 13px;
    line-height: 1.2;
    margin-bottom: 6px;
}

.score-value {
    color: #181813;
    font-size: 30px;
    line-height: 1;
    font-weight: 700;
    letter-spacing: 0;
}

.score-meta {
    color: #64645d;
    font-size: 12px;
    line-height: 1.3;
    margin-top: 6px;
}

.score-compare {
    color: #3f3f39;
    font-size: 12px;
    line-height: 1.3;
    margin-top: 8px;
}

.score-compare.primary {
    color: #1f5132;
}

.score-compare.right {
    color: #1f5132;
}

.score-compare.left {
    color: #6f251c;
}

.score-error {
    background: #fff4f2;
    border: 1px solid #e4b3aa;
    border-radius: 8px;
    color: #5f2118;
    padding: 12px;
}

#nearest-panel {
    margin-top: 6px;
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 8px !important;
    }

    #title {
        display: none;
    }

    #app-shell {
        gap: 8px;
    }

    #score-panel {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
    }

    .score-card {
        padding: 10px;
    }

    .score-label {
        font-size: 12px;
    }

    .score-value {
        font-size: 22px;
    }

    .score-compare {
        font-size: 11px;
    }

    #nearest-panel {
        display: none;
    }
}
"""


def analyze(image_path: str) -> tuple[str, str]:
    if not image_path:
        return empty_scores(), ""

    try:
        store = cached_store()
        embedding = get_face_embedding(
            image_path,
            model_name=store.model_name,
            provider=store.provider,
            det_size=store.det_size,
            det_thresh=store.det_thresh,
            enforce_detection=True,
        )
        result = score_embedding(embedding, store.embeddings, store.ratings, k=TOP_K)
        regressor_text = "Not trained"
        regressor_rating = None
        regressor = cached_regressor(REGRESSOR_PATH)
        if regressor is not None:
            regressor_rating = predict_rating(regressor, embedding)
            regressor_text = regressor.model_name
        multimodal_text = "Not trained"
        multimodal_rating = None
        clip_embedding = None
        multimodal = cached_regressor(MULTIMODAL_REGRESSOR_PATH)
        if multimodal is not None:
            try:
                clip_model = multimodal.metadata.get("clip_model", "openai/clip-vit-base-patch32")
                clip_embedding = get_clip_embedding(image_path, model_name=clip_model, provider="auto")
                multimodal_rating = predict_multimodal_rating(
                    multimodal,
                    face_embedding=embedding,
                    clip_embedding=clip_embedding,
                )
                multimodal_text = multimodal.model_name
            except Exception:
                multimodal_text = "Unavailable"
        original_face_biased = comparison_face_biased_score(
            embedding,
            clip_embedding,
            face_regressor_path=ORIGINAL_REGRESSOR_PATH,
            multimodal_regressor_path=ORIGINAL_MULTIMODAL_REGRESSOR_PATH,
        )
        round1_face_biased = comparison_face_biased_score(
            embedding,
            clip_embedding,
            face_regressor_path=ROUND1_REGRESSOR_PATH,
            multimodal_regressor_path=ROUND1_MULTIMODAL_REGRESSOR_PATH,
        )
    except Exception as exc:
        return error_scores(str(exc)), ""

    nearest_lines = []
    for index, similarity in zip(result.nearest_indices, result.nearest_similarities):
        nearest_lines.append(
            f"- {Path(store.paths[index]).name}: rating {store.ratings[index]:.0f}, similarity {similarity:.4f}"
        )

    nearest_text = "\n".join(nearest_lines)
    details = (
        f"### Nearest references\n{nearest_text}\n\n"
        f"Max similarity: `{result.max_similarity:.4f}`  \n"
        f"Mean similarity: `{result.mean_similarity:.4f}`  \n"
        f"References: `{len(store.embeddings)}`"
    )
    return (
        score_cards(
            knn_rating=result.rating,
            regressor_rating=regressor_rating,
            regressor_name=regressor_text,
            multimodal_rating=multimodal_rating,
            multimodal_name=multimodal_text,
            original_face_biased=original_face_biased,
            round1_face_biased=round1_face_biased,
            max_similarity=result.max_similarity,
        ),
        details,
    )


@lru_cache(maxsize=1)
def cached_store():
    return load_store(STORE_PATH)


@lru_cache(maxsize=4)
def cached_regressor(path: str):
    if not Path(path).exists():
        return None
    return load_regressor(path)


def empty_scores() -> str:
    return f"""
    <div id="score-panel">
      <div class="score-card">
        <div class="score-label">Decision</div>
        <div class="score-value">--</div>
        <div class="score-meta">Threshold {SWIPE_THRESHOLD:.2f}</div>
      </div>
      <div class="score-card">
        <div class="score-label">Face-Biased</div>
        <div class="score-value">--</div>
        <div class="score-meta">22% Ridge + 78% multi</div>
      </div>
      <div class="score-card">
        <div class="score-label">7k Face-Bias</div>
        <div class="score-value">--</div>
        <div class="score-meta">22% Ridge + 78% multi</div>
      </div>
      <div class="score-card">
        <div class="score-label">R1 Face-Bias</div>
        <div class="score-value">--</div>
        <div class="score-meta">22% Ridge + 78% multi</div>
      </div>
      <div class="score-card">
        <div class="score-label">Multimodal</div>
        <div class="score-value">--</div>
        <div class="score-meta">CLIP + face</div>
      </div>
      <div class="score-card">
        <div class="score-label">Ridge</div>
        <div class="score-value">--</div>
        <div class="score-meta">Upload image</div>
      </div>
      <div class="score-card">
        <div class="score-label">KNN</div>
        <div class="score-value">--</div>
        <div class="score-meta">K = 20</div>
      </div>
    </div>
    """


def error_scores(message: str) -> str:
    return f'<div class="score-error">{message}</div>'


def score_cards(
    *,
    knn_rating: float,
    regressor_rating: float | None,
    regressor_name: str,
    multimodal_rating: float | None,
    multimodal_name: str,
    original_face_biased: float | None,
    round1_face_biased: float | None,
    max_similarity: float,
) -> str:
    ridge_value = "--" if regressor_rating is None else f"{regressor_rating:.1f}"
    multimodal_value = "--" if multimodal_rating is None else f"{multimodal_rating:.1f}"
    biased_rating = biased_multimodal_score(regressor_rating, multimodal_rating)
    biased_value = "--" if biased_rating is None else f"{biased_rating:.1f}"
    biased_compare = comparison_text(biased_rating, regressor_rating)
    decision_value, decision_compare, decision_class = decision_text(biased_rating)
    original_value = "--" if original_face_biased is None else f"{original_face_biased:.1f}"
    round1_value = "--" if round1_face_biased is None else f"{round1_face_biased:.1f}"
    original_compare = comparison_text(original_face_biased, biased_rating)
    round1_compare = comparison_text(round1_face_biased, biased_rating)
    multimodal_compare = comparison_text(multimodal_rating, regressor_rating)
    knn_compare = comparison_text(knn_rating, regressor_rating)
    return f"""
    <div id="score-panel">
      <div class="score-card">
        <div class="score-label">Decision</div>
        <div class="score-value">{decision_value}</div>
        <div class="score-meta">Threshold {SWIPE_THRESHOLD:.2f}</div>
        <div class="score-compare {decision_class}">{decision_compare}</div>
      </div>
      <div class="score-card">
        <div class="score-label">Face-Biased</div>
        <div class="score-value">{biased_value}</div>
        <div class="score-meta">{BIASED_FACE_WEIGHT:.0%} Ridge + {1 - BIASED_FACE_WEIGHT:.0%} multi</div>
        <div class="score-compare">{biased_compare}</div>
      </div>
      <div class="score-card">
        <div class="score-label">7k Face-Bias</div>
        <div class="score-value">{original_value}</div>
        <div class="score-meta">22% Ridge + 78% multi</div>
        <div class="score-compare">{original_compare}</div>
      </div>
      <div class="score-card">
        <div class="score-label">R1 Face-Bias</div>
        <div class="score-value">{round1_value}</div>
        <div class="score-meta">22% Ridge + 78% multi</div>
        <div class="score-compare">{round1_compare}</div>
      </div>
      <div class="score-card">
        <div class="score-label">Multimodal</div>
        <div class="score-value">{multimodal_value}</div>
        <div class="score-meta">{multimodal_name}</div>
        <div class="score-compare">{multimodal_compare}</div>
      </div>
      <div class="score-card">
        <div class="score-label">Ridge</div>
        <div class="score-value">{ridge_value}</div>
        <div class="score-meta">{regressor_name}</div>
        <div class="score-compare primary">baseline comparison</div>
      </div>
      <div class="score-card">
        <div class="score-label">KNN</div>
        <div class="score-value">{knn_rating:.1f}</div>
        <div class="score-meta">K = {TOP_K} | max {max_similarity:.3f}</div>
        <div class="score-compare">{knn_compare}</div>
      </div>
    </div>
    """


def biased_multimodal_score(
    regressor_rating: float | None,
    multimodal_rating: float | None,
    *,
    face_weight: float = BIASED_FACE_WEIGHT,
) -> float | None:
    if regressor_rating is None or multimodal_rating is None:
        return None
    return face_weight * regressor_rating + (1 - face_weight) * multimodal_rating


def comparison_face_biased_score(
    embedding,
    clip_embedding,
    *,
    face_regressor_path: str,
    multimodal_regressor_path: str,
) -> float | None:
    if clip_embedding is None:
        return None
    face_regressor = cached_regressor(face_regressor_path)
    multimodal_regressor = cached_regressor(multimodal_regressor_path)
    if face_regressor is None or multimodal_regressor is None:
        return None
    face_rating = predict_rating(face_regressor, embedding)
    multimodal_rating = predict_multimodal_rating(
        multimodal_regressor,
        face_embedding=embedding,
        clip_embedding=clip_embedding,
    )
    return biased_multimodal_score(
        face_rating,
        multimodal_rating,
        face_weight=COMPARISON_FACE_WEIGHT,
    )


def comparison_text(score: float | None, ridge_score: float | None) -> str:
    if score is None or ridge_score is None:
        return "no comparison"
    delta = score - ridge_score
    if abs(delta) < 0.05:
        return "same as Ridge"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{abs(delta):.1f} vs Ridge"


def decision_text(score: float | None) -> tuple[str, str, str]:
    if score is None:
        return "--", "no score", ""
    if score >= SWIPE_THRESHOLD:
        return "RIGHT", f"+{score - SWIPE_THRESHOLD:.1f} over threshold", "right"
    return "LEFT", f"{SWIPE_THRESHOLD - score:.1f} under threshold", "left"


with gr.Blocks(title="BumbleClaw") as demo:
    with gr.Column(elem_id="app-shell"):
        gr.Markdown("# BumbleClaw", elem_id="title")
        with gr.Row():
            with gr.Column(scale=3):
                image_input = gr.Image(
                    type="filepath",
                    label="Upload",
                    show_label=False,
                    height=520,
                    elem_id="image-input",
                )
            with gr.Column(scale=2):
                scores = gr.HTML(value=empty_scores())
                details = gr.Markdown(elem_id="nearest-panel")

        image_input.change(analyze, inputs=image_input, outputs=[scores, details])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT, share=False, css=CSS)
