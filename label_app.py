from __future__ import annotations

import argparse
import random
from pathlib import Path
import socket
from typing import Any

import gradio as gr

from face_similarity.labeling import discover_images, labeled_paths, next_unlabeled_path, upsert_label

SOURCE_DIRS = [
    Path("references") / "Female Faces",
    Path("references") / "women",
    Path("references") / "archive (4)" / "Data_all",
    Path("references") / "Selfie",
    Path("references") / "Selfie-version2",
    Path("references") / "Kpop_Profile_curated",
]
OUTPUT_CSV = Path("dataset_labels.csv")
SERVER_PORT = 7861

MOBILE_CSS = """
:root {
    --bg-color: #000000;
    --panel-color: #1c1c1e;
    --btn-color: #2c2c2e;
    --btn-active-color: #3a3a3c;
    --text-primary: #ffffff;
    --text-muted: #8e8e93;
    --accent-blue: #0a84ff;
    --border-color: #2c2c2e;
}

body,
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    background-color: var(--bg-color) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    max-width: 500px !important;
    margin: 0 auto !important;
    padding: 16px 16px max(16px, env(safe-area-inset-bottom)) !important;
    box-sizing: border-box !important;
}

#label-title {
    margin: 4px 0 6px 0 !important;
    text-align: center !important;
}

#label-title h1 {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: var(--text-muted) !important;
    line-height: 1.2 !important;
    margin: 0 !important;
}

#progress-output,
#filename-output {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    line-height: 14px !important;
    margin-bottom: 4px !important;
    text-align: center !important;
}

#filename-output {
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
    margin-bottom: 6px !important;
}

#progress-output p,
#filename-output p {
    margin: 0 !important;
    font-size: inherit !important;
    color: inherit !important;
}

#image-output {
    border: none !important;
    background-color: var(--panel-color) !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    margin-bottom: 0px !important; /* Flush transition to buttons */
    width: 100% !important;
    aspect-ratio: 1 / 1 !important;
}

#image-output .image-container,
#image-output img {
    width: 100% !important;
    height: 100% !important;
    max-height: none !important;
    object-fit: contain !important;
    aspect-ratio: 1 / 1 !important;
    border-radius: 20px !important;
    background-color: var(--panel-color) !important;
}

#rating-controls {
    padding: 0px 0 max(8px, env(safe-area-inset-bottom)) !important; /* No top padding to keep flush */
    margin: 0 !important;
}

#rating-grid {
    display: grid !important;
    grid-template-columns: repeat(5, 1fr) !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

#rating-grid > div {
    min-width: 0 !important;
    margin: 0 !important;
}

.rate-btn,
.skip-btn {
    -webkit-appearance: none !important;
    appearance: none !important;
    -webkit-tap-highlight-color: transparent !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: -0.2px !important;
    transition: transform 0.08s cubic-bezier(0.25, 1, 0.5, 1) !important;
    cursor: pointer !important;
}

.rate-btn {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    min-height: 60px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    white-space: pre-line !important; /* Preserves vertical newline for layout */
    line-height: 1.25 !important;
}

.rate-btn:active,
.skip-btn:active {
    transform: scale(0.94) !important;
}

.rate-1 {
    background-color: rgba(255, 69, 58, 0.15) !important;
    color: #ff453a !important;
}
.rate-1:active {
    background-color: rgba(255, 69, 58, 0.28) !important;
}

.rate-2 {
    background-color: rgba(255, 159, 10, 0.15) !important;
    color: #ff9f0a !important;
}
.rate-2:active {
    background-color: rgba(255, 159, 10, 0.28) !important;
}

.rate-3 {
    background-color: rgba(142, 142, 147, 0.15) !important;
    color: #a2a2a7 !important;
}
.rate-3:active {
    background-color: rgba(142, 142, 147, 0.28) !important;
}

.rate-4 {
    background-color: rgba(10, 132, 255, 0.15) !important;
    color: #0a84ff !important;
}
.rate-4:active {
    background-color: rgba(10, 132, 255, 0.28) !important;
}

.rate-5 {
    background-color: rgba(48, 209, 88, 0.15) !important;
    color: #30d158 !important;
}
.rate-5:active {
    background-color: rgba(48, 209, 88, 0.28) !important;
}

#skip-button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 !important;
}

.skip-btn {
    min-height: 48px !important;
    background-color: transparent !important;
    color: var(--accent-blue) !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    letter-spacing: -0.4px !important;
    width: 100% !important;
}

@media (max-width: 430px) {
    .gradio-container {
        padding: 8px 8px max(12px, env(safe-area-inset-bottom)) !important;
    }



    #rating-grid {
        gap: 4px !important;
    }

    .rate-btn {
        min-height: 50px !important;
        font-size: 11px !important;
    }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mobile-friendly 1-5 image labeling app.")
    parser.add_argument(
        "--source-dir",
        action="append",
        dest="source_dirs",
        help="Image folder to label. Repeat for multiple folders. Defaults to the reference dataset folders.",
    )
    parser.add_argument("--output-csv", default=str(OUTPUT_CSV), help="CSV to write labels into")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Gradio server port")
    return parser.parse_args()


def configure(*, source_dirs: list[str] | None = None, output_csv: str | Path = OUTPUT_CSV, port: int = SERVER_PORT) -> None:
    global SOURCE_DIRS, OUTPUT_CSV, SERVER_PORT
    if source_dirs:
        SOURCE_DIRS = [Path(path) for path in source_dirs]
    OUTPUT_CSV = Path(output_csv)
    SERVER_PORT = port


def initialize() -> tuple[dict[str, Any], str | None, str, str]:
    paths = discover_all_images(SOURCE_DIRS)
    labels = labeled_paths(OUTPUT_CSV)
    unlabeled = [path for path in paths if path not in labels]
    random.shuffle(unlabeled)

    state = {
        "paths": paths,
        "queue": unlabeled,
        "skipped": [],
    }
    return render(state)


def discover_all_images(source_dirs: list[Path]) -> list[str]:
    paths = []
    for source_dir in source_dirs:
        if source_dir.exists():
            paths.extend(discover_images(source_dir))
    return sorted(set(paths))


def rate_current(state: dict[str, Any] | None, rating_1_5: int) -> tuple[dict[str, Any], str | None, str, str]:
    state = ensure_state(state)
    current = current_path(state)
    if current is not None:
        upsert_label(OUTPUT_CSV, current, rating_1_5)
    return render(state)


def skip_current(state: dict[str, Any] | None) -> tuple[dict[str, Any], str | None, str, str]:
    state = ensure_state(state)
    current = current_path(state)
    if current is not None and current not in state["skipped"]:
        state["skipped"].append(current)
    return render(state)


def render(state: dict[str, Any]) -> tuple[dict[str, Any], str | None, str, str]:
    current = current_path(state)
    labels = labeled_paths(OUTPUT_CSV)
    total = len(state["paths"])
    skipped = len(state["skipped"])
    remaining = max(total - len(labels) - skipped, 0)

    if current is None:
        filename = "All available images are labeled or skipped for this session."
        image_path = None
    else:
        filename = f"Current: `{current}`"
        image_path = current

    progress = (
        f"Labeled: `{len(labels)}` / `{total}` | "
        f"Remaining: `{remaining}` | "
        f"Skipped this session: `{skipped}` | "
        f"Output: `{OUTPUT_CSV.as_posix()}`"
    )
    return state, image_path, filename, progress


def current_path(state: dict[str, Any]) -> str | None:
    return next_unlabeled_path(
        state.get("queue", state["paths"]),
        labeled_paths(OUTPUT_CSV),
        set(state["skipped"]),
    )


def ensure_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if state and "paths" in state and "queue" in state and "skipped" in state:
        return state
    initialized, _, _, _ = initialize()
    return initialized


def lan_urls(port: int) -> list[str]:
    urls = []
    try:
        hostname = socket.gethostname()
        addresses = socket.gethostbyname_ex(hostname)[2]
    except OSError:
        addresses = []

    for address in sorted(set(addresses)):
        if not address.startswith("127."):
            urls.append(f"http://{address}:{port}")
    return urls


def print_lan_urls(port: int) -> None:
    urls = lan_urls(port)
    if not urls:
        print("Open this app from your phone using your computer's Wi-Fi IP address.")
        return

    print("\nOpen on iPhone Safari while connected to the same Wi-Fi:")
    for url in urls:
        print(f"  {url}")
    print()


def allowed_paths() -> list[str]:
    return [str(path.resolve()) for path in SOURCE_DIRS if path.exists()]


with gr.Blocks(title="Dataset Labeler") as demo:
    app_state = gr.State()
    image_output = gr.Image(label=None, interactive=False, elem_id="image-output")

    with gr.Column(elem_id="rating-controls"):
        with gr.Row(elem_id="rating-grid"):
            btn_1 = gr.Button("1\nNo", elem_classes=["rate-btn", "rate-1"])
            btn_2 = gr.Button("2\nDislike", elem_classes=["rate-btn", "rate-2"])
            btn_3 = gr.Button("3\nNeutral", elem_classes=["rate-btn", "rate-3"])
            btn_4 = gr.Button("4\nLike", elem_classes=["rate-btn", "rate-4"])
            btn_5 = gr.Button("5\nLove", elem_classes=["rate-btn", "rate-5"])

        skip_button = gr.Button("Skip", elem_id="skip-button", elem_classes=["skip-btn"])

        filename_output = gr.Markdown(elem_id="filename-output")
        progress_output = gr.Markdown(elem_id="progress-output")
        gr.Markdown("# Dataset Labeler", elem_id="label-title")

    outputs = [app_state, image_output, filename_output, progress_output]
    demo.load(initialize, outputs=outputs)
    btn_1.click(lambda state: rate_current(state, 1), inputs=app_state, outputs=outputs, scroll_to_output=False, show_progress="hidden")
    btn_2.click(lambda state: rate_current(state, 2), inputs=app_state, outputs=outputs, scroll_to_output=False, show_progress="hidden")
    btn_3.click(lambda state: rate_current(state, 3), inputs=app_state, outputs=outputs, scroll_to_output=False, show_progress="hidden")
    btn_4.click(lambda state: rate_current(state, 4), inputs=app_state, outputs=outputs, scroll_to_output=False, show_progress="hidden")
    btn_5.click(lambda state: rate_current(state, 5), inputs=app_state, outputs=outputs, scroll_to_output=False, show_progress="hidden")
    skip_button.click(skip_current, inputs=app_state, outputs=outputs, scroll_to_output=False, show_progress="hidden")


if __name__ == "__main__":
    cli_args = parse_args()
    configure(source_dirs=cli_args.source_dirs, output_csv=cli_args.output_csv, port=cli_args.port)
    print_lan_urls(SERVER_PORT)
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT, share=False, css=MOBILE_CSS, allowed_paths=allowed_paths())
