from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

DEFAULT_DATA_DIR = Path(r"D:\BumbleLog")
DEFAULT_CSV = DEFAULT_DATA_DIR / "scores.csv"
SERVER_PORT = int(os.environ.get("BUMBLE_FINDER_PORT", "7862"))
CACHE_SCHEMA_VERSION = 2

CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}

#finder-title h1 {
    font-size: 24px;
    line-height: 1.2;
    margin: 8px 0 0;
    letter-spacing: 0;
}

#status-output {
    color: #4a4a45;
    font-size: 13px;
}

#preview-image {
    border-radius: 8px;
    overflow: hidden;
}
"""


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def resolve_image_path(row: dict[str, str], data_dir: Path) -> Path:
    screenshot = row.get("screenshot", "").strip()
    if not screenshot:
        return data_dir / "__missing_screenshot__"

    path = Path(screenshot)
    if path.is_absolute():
        return path
    return data_dir / path


def load_rows(csv_path: str, data_dir: str) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(csv_path).expanduser()
    image_dir = Path(data_dir).expanduser()

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])

    for row in rows:
        image_path = resolve_image_path(row, image_dir)
        row["_image_path"] = str(image_path)
        row["_exists"] = "yes" if image_path.exists() else "no"
    return rows, columns


def row_matches(
    row: dict[str, str],
    query: str,
    action: str,
    min_score: float | None,
    max_score: float | None,
) -> bool:
    if query:
        haystack = " ".join(
            [
                row.get("timestamp", ""),
                row.get("screenshot", ""),
                row.get("method", ""),
                row.get("action", ""),
            ]
        ).lower()
        if query.lower() not in haystack:
            return False

    if action != "any" and row.get("action", "").strip().lower() != action:
        return False

    score = parse_float(row.get("score"))
    if min_score is not None and (score is None or score < min_score):
        return False
    if max_score is not None and (score is None or score > max_score):
        return False

    return True


def sort_rows(rows: list[dict[str, str]], sort_mode: str) -> list[dict[str, str]]:
    if sort_mode == "score asc":
        return sorted(
            rows,
            key=lambda row: parse_float(row.get("score"))
            if parse_float(row.get("score")) is not None
            else float("inf"),
        )
    if sort_mode == "oldest":
        return sorted(rows, key=lambda row: row.get("timestamp", ""))
    if sort_mode == "newest":
        return sorted(rows, key=lambda row: row.get("timestamp", ""), reverse=True)
    return sorted(
        rows,
        key=lambda row: parse_float(row.get("score"))
        if parse_float(row.get("score")) is not None
        else float("-inf"),
        reverse=True,
    )


def display_value(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    number = parse_float(value)
    if number is None:
        return value
    return f"{number:.2f}"


def table_for(rows: list[dict[str, str]], columns: list[str]) -> pd.DataFrame:
    table_rows = []
    has_similarity = any("_similarity" in row for row in rows)
    for index, row in enumerate(rows):
        table_row = {"#": index}
        if has_similarity:
            table_row["similarity"] = row.get("_similarity", "")
        for column in columns:
            table_row[column] = row.get(column, "")
        table_row["exists"] = row.get("_exists", "no")
        table_row["image_path"] = row.get("_image_path", "")
        table_rows.append(table_row)

    output_columns = ["#"]
    if has_similarity:
        output_columns.append("similarity")
    output_columns.extend([*columns, "exists", "image_path"])
    return pd.DataFrame(table_rows, columns=output_columns)


def image_for_index(rows: list[dict[str, str]], index: int) -> tuple[str | None, str]:
    if not rows:
        return None, "No matching rows."

    index = max(0, min(index, len(rows) - 1))
    row = rows[index]
    image_path = Path(row.get("_image_path", ""))
    score = display_value(row, "score")
    caption = (
        f"{index}: {row.get('screenshot', '')} | "
        f"action={row.get('action', '')} | score={score} | exists={row.get('_exists', 'no')}"
    )

    if not image_path.exists():
        return None, f"{caption}\nMissing file: {image_path}"
    return str(image_path), caption


def search(
    csv_path: str,
    data_dir: str,
    query: str,
    action: str,
    min_score: str | float | None,
    max_score: str | float | None,
    sort_mode: str,
    limit: int,
) -> tuple[pd.DataFrame, str, str | None, str, list[dict[str, str]], int]:
    try:
        rows, columns = load_rows(csv_path, data_dir)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read CSV: {exc}", None, "", [], 0

    filtered = [
        row
        for row in rows
        if row_matches(
            row=row,
            query=(query or "").strip(),
            action=action,
            min_score=parse_float(min_score),
            max_score=parse_float(max_score),
        )
    ]
    sorted_rows = sort_rows(filtered, sort_mode)
    row_limit = int(limit or 0)
    visible_rows = sorted_rows if row_limit <= 0 else sorted_rows[:row_limit]
    image_path, caption = image_for_index(visible_rows, 0)
    status = f"Showing {len(visible_rows)} of {len(filtered)} matches from {len(rows)} rows."
    return table_for(visible_rows, columns), status, image_path, caption, visible_rows, 0


def cache_path_for(data_dir: str) -> Path:
    return Path(data_dir).expanduser() / ".picture_finder_features.npz"


def image_feature(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        full = image.resize((24, 24), Image.Resampling.BILINEAR)
        crop = ImageOps.fit(image, (24, 24), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
        full_array = np.asarray(full, dtype=np.float32)
        crop_array = np.asarray(crop, dtype=np.float32)

        parts = [
            full_array.reshape(-1) / 255.0,
            crop_array.reshape(-1) / 255.0,
        ]
        for channel in range(3):
            hist, _ = np.histogram(full_array[:, :, channel], bins=16, range=(0, 256))
            hist = hist.astype(np.float32)
            total = float(hist.sum())
            parts.append(hist / total if total else hist)

    feature = np.concatenate(parts).astype(np.float32)
    feature -= float(feature.mean())
    norm = float(np.linalg.norm(feature))
    if norm:
        feature /= norm
    return feature


def unique_existing_image_paths(rows: list[dict[str, str]]) -> list[str]:
    paths = []
    seen = set()
    for row in rows:
        image_path = row.get("_image_path", "")
        if image_path in seen or row.get("_exists") != "yes":
            continue
        seen.add(image_path)
        paths.append(image_path)
    return paths


def load_feature_cache(cache_path: Path) -> dict[str, tuple[float, int, np.ndarray]]:
    if not cache_path.exists():
        return {}

    try:
        data = np.load(cache_path, allow_pickle=False)
        if int(data.get("schema_version", 0)) != CACHE_SCHEMA_VERSION:
            return {}
        paths = data["paths"]
        mtimes = data["mtimes"]
        sizes = data["sizes"]
        features = data["features"]
    except Exception:
        return {}

    cache = {}
    for index, image_path in enumerate(paths):
        cache[str(image_path)] = (float(mtimes[index]), int(sizes[index]), features[index])
    return cache


def cached_feature_is_current(
    cached: tuple[float, int, np.ndarray] | None,
    stat: os.stat_result,
) -> bool:
    return cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size


def compute_feature_item(item: tuple[str, float, int]) -> tuple[str, float, int, np.ndarray] | None:
    image_path, mtime, size = item
    try:
        return image_path, mtime, size, image_feature(image_path)
    except Exception:
        return None


def save_feature_cache(cache_path: Path, cache: dict[str, tuple[float, int, np.ndarray]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    paths = np.asarray(list(cache.keys()), dtype=str)
    mtimes = np.asarray([item[0] for item in cache.values()], dtype=np.float64)
    sizes = np.asarray([item[1] for item in cache.values()], dtype=np.int64)
    features = np.asarray([item[2] for item in cache.values()], dtype=np.float32)
    np.savez_compressed(
        cache_path,
        schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        paths=paths,
        mtimes=mtimes,
        sizes=sizes,
        features=features,
    )


def feature_matrix_for(rows: list[dict[str, str]], data_dir: str) -> tuple[list[str], np.ndarray, int]:
    image_paths = unique_existing_image_paths(rows)
    cache_path = cache_path_for(data_dir)
    cache = load_feature_cache(cache_path)
    kept_cache = {}
    missing = []
    features = []
    valid_paths = []

    for image_path in image_paths:
        path = Path(image_path)
        try:
            stat = path.stat()
        except OSError:
            continue

        cached = cache.get(image_path)
        if cached_feature_is_current(cached, stat):
            kept_cache[image_path] = cached
            valid_paths.append(image_path)
            features.append(cached[2])
        else:
            missing.append((image_path, float(stat.st_mtime), int(stat.st_size)))

    worker_count = min(16, max(1, (os.cpu_count() or 4)))
    if missing:
        computed_by_path = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(compute_feature_item, item) for item in missing]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                image_path, mtime, size, feature = result
                computed_by_path[image_path] = (mtime, size, feature)

        for image_path, _mtime, _size in missing:
            computed = computed_by_path.get(image_path)
            if computed is None:
                continue
            kept_cache[image_path] = computed
            valid_paths.append(image_path)
            features.append(computed[2])

    if missing or len(kept_cache) != len(cache):
        save_feature_cache(cache_path, kept_cache)

    if not features:
        return [], np.empty((0, 0), dtype=np.float32), len(missing)
    return valid_paths, np.vstack(features).astype(np.float32), len(missing)


def find_similar(
    uploaded_image: str | None,
    csv_path: str,
    data_dir: str,
    query: str,
    action: str,
    min_score: str | float | None,
    max_score: str | float | None,
    top_n: int,
) -> tuple[pd.DataFrame, str, str | None, str, list[dict[str, str]], int]:
    if not uploaded_image:
        return pd.DataFrame(), "Upload an image first.", None, "", [], 0

    try:
        rows, columns = load_rows(csv_path, data_dir)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read CSV: {exc}", None, "", [], 0

    filtered_rows = [
        row
        for row in rows
        if row_matches(
            row=row,
            query=(query or "").strip(),
            action=action,
            min_score=parse_float(min_score),
            max_score=parse_float(max_score),
        )
    ]
    paths, features, updated = feature_matrix_for(filtered_rows, data_dir)
    if len(paths) == 0:
        return pd.DataFrame(), "No readable images found in the current filtered rows.", None, "", [], 0

    try:
        query_feature = image_feature(uploaded_image)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read uploaded image: {exc}", None, "", [], 0

    similarities = features @ query_feature
    count = max(1, min(int(top_n or 1), len(paths)))
    top_indices = np.argsort(-similarities)[:count]

    rows_by_path = {}
    for row in filtered_rows:
        rows_by_path.setdefault(row.get("_image_path", ""), row)

    ranked_rows = []
    for index in top_indices:
        image_path = paths[int(index)]
        row = dict(rows_by_path[image_path])
        row["_similarity"] = f"{float(similarities[int(index)]):.4f}"
        ranked_rows.append(row)

    image_path, caption = image_for_index(ranked_rows, 0)
    status = (
        f"Found top {len(ranked_rows)} similar images from {len(paths)} indexed images "
        f"({len(filtered_rows)} filtered CSV rows, {updated} new/changed image features cached)."
    )
    return table_for(ranked_rows, columns), status, image_path, caption, ranked_rows, 0


def initial_state() -> tuple[pd.DataFrame, str, None, str, list[dict[str, str]], int]:
    return (
        pd.DataFrame(),
        "Upload an image and click Find Similar. Use Show Logs when you want the full CSV table.",
        None,
        "",
        [],
        0,
    )


def show_selected(rows: list[dict[str, str]], index: float | int | None) -> tuple[str | None, str, int]:
    selected_index = int(index or 0)
    image_path, caption = image_for_index(rows, selected_index)
    clamped_index = max(0, min(selected_index, max(len(rows) - 1, 0)))
    return image_path, caption, clamped_index


def move_selection(rows: list[dict[str, str]], index: float | int | None, delta: int) -> tuple[str | None, str, int]:
    return show_selected(rows, int(index or 0) + delta)


def open_selected(rows: list[dict[str, str]], index: float | int | None) -> str:
    selected_index = max(0, min(int(index or 0), max(len(rows) - 1, 0)))
    if not rows:
        return "No matching row to open."

    image_path = Path(rows[selected_index].get("_image_path", ""))
    if not image_path.exists():
        return f"Missing file: {image_path}"

    os.startfile(image_path)
    return f"Opened {image_path}"


def build_app(default_csv: Path, default_data_dir: Path) -> gr.Blocks:
    with gr.Blocks(title="BumbleLog Picture Finder") as demo:
        gr.Markdown("# BumbleLog Picture Finder", elem_id="finder-title")
        rows_state = gr.State([])

        with gr.Row():
            csv_path = gr.Textbox(label="scores.csv", value=str(default_csv))
            data_dir = gr.Textbox(label="image folder", value=str(default_data_dir))

        with gr.Row():
            query = gr.Textbox(label="filename / timestamp search", placeholder="profile_20260520 or 181814")
            action = gr.Dropdown(["any", "right", "left"], value="any", label="action")
            min_score = gr.Textbox(label="min score", placeholder="blank = no minimum")
            max_score = gr.Textbox(label="max score", placeholder="blank = no maximum")

        with gr.Row():
            sort_mode = gr.Dropdown(
                ["score desc", "score asc", "newest", "oldest"],
                value="score desc",
                label="sort",
            )
            limit = gr.Number(label="max rows (0 = all)", value=0, precision=0)
            top_n = gr.Number(label="similar matches", value=25, precision=0)
            selected_index = gr.Number(label="selected #", value=0, precision=0)

        uploaded_image = gr.Image(label="upload image to find", type="filepath")

        with gr.Row():
            similar_button = gr.Button("Find Similar", variant="primary")
            search_button = gr.Button("Show Logs")
            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next")
            open_button = gr.Button("Open Image")

        status = gr.Markdown(elem_id="status-output")
        open_status = gr.Textbox(label="open status", interactive=False)

        with gr.Row():
            table = gr.Dataframe(label="matches", interactive=False, wrap=True)
            preview = gr.Image(label="preview", type="filepath", elem_id="preview-image")

        caption = gr.Textbox(label="selected image", interactive=False)

        search_inputs = [
            csv_path,
            data_dir,
            query,
            action,
            min_score,
            max_score,
            sort_mode,
            limit,
        ]
        search_outputs = [table, status, preview, caption, rows_state, selected_index]
        search_button.click(search, inputs=search_inputs, outputs=search_outputs)
        demo.load(initial_state, outputs=search_outputs)

        similar_inputs = [
            uploaded_image,
            csv_path,
            data_dir,
            query,
            action,
            min_score,
            max_score,
            top_n,
        ]
        similar_button.click(find_similar, inputs=similar_inputs, outputs=search_outputs)

        selected_index.change(
            show_selected,
            inputs=[rows_state, selected_index],
            outputs=[preview, caption, selected_index],
        )
        previous_button.click(
            move_selection,
            inputs=[rows_state, selected_index, gr.State(-1)],
            outputs=[preview, caption, selected_index],
        )
        next_button.click(
            move_selection,
            inputs=[rows_state, selected_index, gr.State(1)],
            outputs=[preview, caption, selected_index],
        )
        open_button.click(open_selected, inputs=[rows_state, selected_index], outputs=open_status)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse pictures referenced by a BumbleLog scores.csv file.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to scores.csv.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Folder containing screenshot files.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port.")
    args = parser.parse_args()

    app = build_app(default_csv=Path(args.csv), default_data_dir=Path(args.data_dir))
    app.launch(
        server_name=args.host,
        server_port=args.port,
        css=CSS,
        allowed_paths=[str(Path(args.data_dir).expanduser())],
    )


if __name__ == "__main__":
    main()
