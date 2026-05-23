import os
import json
import numpy as np
import pandas as pd
import gradio as gr
import cv2
from insightface.app import FaceAnalysis

CSV_PATH = r"D:\BumbleLog\scores.csv"
IMAGE_DIR = r"D:\BumbleLog"
HASH_CACHE_PATH = os.path.join(IMAGE_DIR, ".familiar_faces_cache.json")
EMBED_CACHE_PATH = os.path.join(IMAGE_DIR, ".face_embeddings.npz")
FACE_APP = None

def get_face_app():
    global FACE_APP
    if FACE_APP is None:
        FACE_APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
    return FACE_APP

def face_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    app = get_face_app()
    faces = app.get(img)

    if not faces:
        return None

    # use largest detected face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    emb = face.normed_embedding.astype("float32")
    return emb

def load_face_embedding_cache(progress=gr.Progress()):
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".webp", ".jpg", ".png"))]

    if os.path.exists(EMBED_CACHE_PATH):
        data = np.load(EMBED_CACHE_PATH, allow_pickle=True)
        cached_files = list(data["files"])
        cached_embs = data["embs"].astype("float32")
    else:
        cached_files = []
        cached_embs = np.empty((0, 512), dtype="float32")

    cached_set = set(cached_files)
    missing = [f for f in files if f not in cached_set]

    if missing:
        new_files = []
        new_embs = []

        total = len(missing)
        progress(0, desc="Building face embedding cache")

        for i, f in enumerate(missing):
            if i % 50 == 0:
                progress(i / max(total, 1), desc=f"Embedding face {i}/{total}")

            img_path = os.path.join(IMAGE_DIR, f)
            emb = face_embedding(img_path)

            if emb is not None:
                new_files.append(f)
                new_embs.append(emb)

        if new_embs:
            cached_files.extend(new_files)
            if cached_embs.shape[0] == 0:
                cached_embs = np.array(new_embs, dtype="float32")
            else:
                cached_embs = np.vstack([cached_embs, np.array(new_embs, dtype="float32")])

        np.savez_compressed(
            EMBED_CACHE_PATH,
            files=np.array(cached_files),
            embs=cached_embs
        )

        progress(1.0, desc="Face cache complete")

    return cached_files, cached_embs

def dhash(img_path, hash_size=8):
    """Compute difference hash — resize to (hash_size+1, hash_size), compare adjacent pixels."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum(1 << i for i, b in enumerate(diff.flatten()) if b)
    except Exception:
        return None

def hamming(h1, h2):
    if h1 is None or h2 is None: return 999
    return bin(h1 ^ h2).count("1")

def load_hash_cache():
    if os.path.exists(HASH_CACHE_PATH):
        try:
            with open(HASH_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_hash_cache(cache):
    try:
        with open(HASH_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

def load_and_process_data(method="Multi-Model Score Vector"):
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df_clean = df.copy()
    
    if method == "Multi-Model Score Vector":
        req_cols = ["ridge", "knn", "multimodal"]
        if not all(col in df_clean.columns for col in req_cols):
            return df, pd.DataFrame()
            
        df_clean["ridge_round"] = pd.to_numeric(df_clean["ridge"], errors="coerce").round(3)
        df_clean["knn_round"] = pd.to_numeric(df_clean["knn"], errors="coerce").round(3)
        df_clean["multi_round"] = pd.to_numeric(df_clean["multimodal"], errors="coerce").round(3)
        
        df_clean = df_clean.dropna(subset=["ridge_round", "knn_round", "multi_round"])
        
        group_cols = ["ridge_round", "knn_round", "multi_round"]
        df_clean["cluster_id"] = df_clean.groupby(group_cols).ngroup()
        
    elif method == "Ridge & KNN (<0.01 deviation)":
        from sklearn.cluster import DBSCAN
        req_cols = ["ridge", "knn"]
        if not all(col in df_clean.columns for col in req_cols):
            return df, pd.DataFrame()
            
        df_clean = df_clean.dropna(subset=req_cols).copy()
        df_clean["ridge"] = pd.to_numeric(df_clean["ridge"], errors="coerce")
        df_clean["knn"] = pd.to_numeric(df_clean["knn"], errors="coerce")
        df_clean = df_clean.dropna(subset=req_cols)
        
        X = df_clean[["ridge", "knn"]].values
        db = DBSCAN(eps=0.01, min_samples=2, n_jobs=-1).fit(X)
        df_clean["cluster_id"] = db.labels_
        
        # Filter out noise (-1)
        df_clean = df_clean[df_clean["cluster_id"] != -1].copy()
        
    else: # Ridge-only (<0.001 diff)
        if "ridge" not in df_clean.columns:
            return df, pd.DataFrame()
            
        df_clean = df_clean.dropna(subset=["ridge"]).copy()
        df_clean["ridge"] = pd.to_numeric(df_clean["ridge"], errors="coerce")
        df_clean = df_clean.dropna(subset=["ridge"])
        df_clean = df_clean.sort_values(by="ridge").reset_index(drop=True)
        
        df_clean["ridge_diff"] = df_clean["ridge"].diff().abs()
        new_cluster_mask = (df_clean["ridge_diff"] >= 0.001) | (df_clean["ridge_diff"].isna())
        df_clean["cluster_id"] = new_cluster_mask.cumsum()
        
    # We only want clusters with > 1 element
    cluster_counts = df_clean["cluster_id"].value_counts()
    familiar_clusters = cluster_counts[cluster_counts > 1].index
    
    df_familiar = df_clean[df_clean["cluster_id"].isin(familiar_clusters)].copy()
    
    # Re-map cluster_id to clean 1-based sequential integers for UI display
    if not df_familiar.empty:
        df_familiar["cluster_id"] = df_familiar.groupby("cluster_id").ngroup() + 1
        df_familiar = df_familiar.sort_values(by="cluster_id").reset_index(drop=True)
        
    return df, df_familiar

def get_cluster_summary(df_familiar):
    if df_familiar.empty:
        return pd.DataFrame(columns=["Cluster ID", "Images Count", "Mean Ridge Score"])
        
    summary = df_familiar.groupby("cluster_id").agg(
        Images_Count=("screenshot", "count"),
        Mean_Ridge=("ridge", "mean")
    ).reset_index()
    
    summary = summary.rename(columns={
        "cluster_id": "Cluster ID",
        "Images_Count": "Images Count",
        "Mean_Ridge": "Mean Ridge Score"
    })
    
    summary["Mean Ridge Score"] = pd.to_numeric(summary["Mean Ridge Score"], errors="coerce").round(4)
    
    return summary

def make_stats_html(df_total, df_familiar):
    total_logs = len(df_total) if not df_total.empty else 0
    total_clusters = df_familiar["cluster_id"].nunique() if not df_familiar.empty else 0
    max_cluster_size = df_familiar["cluster_id"].value_counts().max() if not df_familiar.empty else 0
    total_matched = len(df_familiar) if not df_familiar.empty else 0
    
    html = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; font-family: 'Inter', sans-serif;">
        <div style="flex: 1; min-width: 180px; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 18px; backdrop-filter: blur(12px); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Total Processed Logs</div>
            <div style="color: #f8fafc; font-size: 1.85rem; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em;">{total_logs:,}</div>
        </div>
        <div style="flex: 1; min-width: 180px; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 18px; backdrop-filter: blur(12px); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);">
            <div style="color: #818cf8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Familiar Face Clusters</div>
            <div style="color: #a5b4fc; font-size: 1.85rem; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em;">{total_clusters:,}</div>
        </div>
        <div style="flex: 1; min-width: 180px; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 18px; backdrop-filter: blur(12px); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);">
            <div style="color: #10b981; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Total Matched Images</div>
            <div style="color: #6ee7b7; font-size: 1.85rem; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em;">{total_matched:,}</div>
        </div>
        <div style="flex: 1; min-width: 180px; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 18px; backdrop-filter: blur(12px); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);">
            <div style="color: #f43f5e; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Largest Cluster Size</div>
            <div style="color: #fda4af; font-size: 1.85rem; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em;">{max_cluster_size:,}</div>
        </div>
    </div>
    """
    return html

def render_cluster(cluster_id, df_familiar):
    if df_familiar.empty or cluster_id is None:
        return [], pd.DataFrame(), ""
        
    cluster_data = df_familiar[df_familiar["cluster_id"] == cluster_id]
    total_images = len(cluster_data)
    
    MAX_GALLERY_IMAGES = 60
    gallery_data = cluster_data.head(MAX_GALLERY_IMAGES)
    
    images = []
    for _, row in gallery_data.iterrows():
        img_path = os.path.join(IMAGE_DIR, str(row["screenshot"]))
        if os.path.exists(img_path):
            score_val = pd.to_numeric(row.get('score', None), errors='coerce')
            ridge_val = pd.to_numeric(row.get('ridge', None), errors='coerce')
            
            if pd.notna(score_val) and pd.notna(ridge_val):
                caption = f"Score: {score_val:.2f} | Ridge: {ridge_val:.4f}"
            elif pd.notna(ridge_val):
                caption = f"Ridge: {ridge_val:.4f}"
            else:
                caption = "N/A"
            images.append((img_path, caption))
            
    details_cols = ["timestamp", "screenshot", "score", "final_score", "ridge", "knn"]
    avail_cols = [c for c in details_cols if c in cluster_data.columns]
    details_df = cluster_data[avail_cols]
    
    badge_html = f"""
    <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; font-family: 'Inter', sans-serif; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
        <div>
            <span style="color: #a5b4fc; font-weight: 700; font-size: 1.05rem;">👯 Cluster #{cluster_id}</span>
            <span style="color: #4b5563; margin: 0 10px;">|</span>
            <span style="color: #e2e8f0; font-size: 0.95rem;">Total Images found: <strong style="color: #c7d2fe;">{total_images}</strong></span>
        </div>
        {f'<div style="background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 6px; padding: 4px 10px; color: #fda4af; font-size: 0.8rem; font-weight: 600;">⚡ Showing first {MAX_GALLERY_IMAGES} for high performance</div>' if total_images > MAX_GALLERY_IMAGES else '<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 4px 10px; color: #a7f3d0; font-size: 0.8rem; font-weight: 600;">⚡ Showing all images</div>'}
    </div>
    """
    
    return images, details_df, badge_html

def on_cluster_select(evt: gr.SelectData, summary_df, df_familiar):
    row_idx = evt.index[0]
    cluster_id = summary_df.iloc[row_idx]["Cluster ID"]
    return render_cluster(cluster_id, df_familiar)

def find_similar_image(uploaded_img_path, df_familiar, summary_df, df_total_state, progress=gr.Progress()):
    if not uploaded_img_path:
        return None, None, "<div style='color: #f43f5e;'>Please upload an image.</div>"

    matched_file = None
    uncertain_note = ""
    
    # 1. Try face embedding search
    query_emb = face_embedding(uploaded_img_path)
    if query_emb is not None:
        files, embs = load_face_embedding_cache(progress)
        if len(files) > 0:
            sims = embs @ query_emb
            top_k = 30
            idxs = np.argsort(-sims)[:top_k]

            best_idx = idxs[0]
            best_file = files[best_idx]
            best_score = float(sims[best_idx])

            second_score = float(sims[idxs[1]]) if len(idxs) > 1 else -1
            margin = best_score - second_score

            MIN_SCORE = 0.35
            MIN_MARGIN = 0.02

            if best_score >= MIN_SCORE:
                if margin < MIN_MARGIN:
                    uncertain_note = f"""
                    <div style="margin-top: 8px; color: #fbbf24; font-size: 0.9rem;">
                        ⚠️ Similar-looking face risk: top match margin is only {margin:.3f}.
                    </div>
                    """
                matched_file = best_file

    # 2. Fallback to dHash if no face match was found
    if matched_file is None:
        try:
            up_hash = dhash(uploaded_img_path)
        except Exception:
            up_hash = None
            
        if up_hash is not None:
            cache = load_hash_cache()
            all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".webp", ".jpg", ".png"))]
            
            if cache is None or len(cache) < len(all_files):
                cache = cache or {}
                files_to_hash = [f for f in all_files if f not in cache]
                total = len(files_to_hash)
                
                progress(0, desc="Building Global Image Hash Cache")
                for i, f in enumerate(files_to_hash):
                    if i % 100 == 0:
                        progress(i / total, desc=f"Hashing image {i}/{total}...")
                    try:
                        img_path = os.path.join(IMAGE_DIR, f)
                        h = dhash(img_path)
                        if h is not None:
                            cache[f] = h
                    except Exception:
                        continue
                save_hash_cache(cache)
                progress(1.0, desc="Hash Cache build complete!")
                
            for f, h in cache.items():
                if hamming(up_hash, h) <= 2:
                    matched_file = f
                    break

    if matched_file is not None:
        # Check if matched_file is in the active familiar clusters
        if not df_familiar.empty:
            cluster_rows = df_familiar[df_familiar["screenshot"] == matched_file]
            if not cluster_rows.empty:
                matched_cluster = cluster_rows.iloc[0]["cluster_id"]
                images, details, badge = render_cluster(matched_cluster, df_familiar)
                new_badge = badge.replace('👯 Cluster', f'🎯 GLOBALLY MATCHED! Displaying Cluster')
                if uncertain_note:
                    new_badge += uncertain_note
                return images, details, new_badge
                
        # If it's in the log, but not in any active cluster -> Render Virtual Cluster
        df_total = df_total_state
        if df_total is not None and not df_total.empty:
            row_data = df_total[df_total["screenshot"] == matched_file]
            if not row_data.empty:
                img_path = os.path.join(IMAGE_DIR, matched_file)
                row = row_data.iloc[0]
                score_val = pd.to_numeric(row.get('score', None), errors='coerce')
                ridge_val = pd.to_numeric(row.get('ridge', None), errors='coerce')
                
                if pd.notna(score_val) and pd.notna(ridge_val):
                    caption = f"Score: {score_val:.2f} | Ridge: {ridge_val:.4f}"
                elif pd.notna(ridge_val):
                    caption = f"Ridge: {ridge_val:.4f}"
                else:
                    caption = "N/A"
                    
                images = [(img_path, caption)]
                details_cols = ["timestamp", "screenshot", "score", "final_score", "ridge", "knn"]
                avail_cols = [c for c in details_cols if c in row_data.columns]
                details_df = row_data[avail_cols]
                
                badge = f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; font-family: 'Inter', sans-serif; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                    <div>
                        <span style="color: #6ee7b7; font-weight: 700; font-size: 1.05rem;">🎯 GLOBAL MATCH: Isolated Profile</span>
                        <span style="color: #4b5563; margin: 0 10px;">|</span>
                        <span style="color: #e2e8f0; font-size: 0.95rem;">Total Images found: <strong style="color: #a7f3d0;">1</strong></span>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 4px 10px; color: #a7f3d0; font-size: 0.8rem; font-weight: 600;">✨ Found in global logs (not in active cluster)</div>
                </div>
                """
                if uncertain_note:
                    badge += uncertain_note
                return images, details_df, badge

    return None, None, f"""
    <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 10px; padding: 16px; margin-bottom: 16px;">
        <strong style="color: #fda4af;">No Match Found</strong><br>
        <span style="color: #e2e8f0; font-size: 0.9rem;">Scanned entire D:\\BumbleLog globally but found no visual or face match.</span>
    </div>
    """

def on_method_change(method):
    df_total, df_familiar = load_and_process_data(method)
    summary = get_cluster_summary(df_familiar)
    stats_html = make_stats_html(df_total, df_familiar)
    return df_total, df_familiar, summary, summary, stats_html, None, pd.DataFrame(), ""


custom_css = """
body {
    background-color: #080c14;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 0%, rgba(139, 92, 246, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(244, 63, 94, 0.05) 0px, transparent 50%);
    background-attachment: fixed;
    margin: 0;
    padding: 0;
}
.gradio-container {
    max-width: 1440px !important;
    padding: 32px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff 30%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px !important;
    letter-spacing: -0.03em !important;
}
h3 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin-top: 0 !important;
    margin-bottom: 12px !important;
    letter-spacing: -0.01em !important;
}
.block {
    background: rgba(17, 24, 39, 0.45) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3) !important;
}
.dataframe th {
    background-color: rgba(99, 102, 241, 0.08) !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}
.dataframe td {
    color: #e2e8f0 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}
.dataframe tr:hover td {
    background-color: rgba(99, 102, 241, 0.05) !important;
    color: #ffffff !important;
}
#gallery {
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    background: rgba(15, 23, 42, 0.3) !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
    block_radius="14px",
)

with gr.Blocks(title="Familiar Faces") as app:
    df_total_state = gr.State()
    df_familiar_state = gr.State()
    summary_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=8):
            gr.HTML(
                """
                <div style="margin-bottom: 24px; font-family: 'Inter', sans-serif;">
                    <h1 style="margin: 0; font-size: 2.25rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">👯 Familiar Faces</h1>
                    <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">Discover and inspect recurring profiles in your Bumble log.</p>
                </div>
                """
            )
        with gr.Column(scale=4):
            method_radio = gr.Radio(
                choices=["Multi-Model Score Vector", "Ridge & KNN (<0.01 deviation)", "Ridge-only (<0.001 diff)"],
                value="Multi-Model Score Vector",
                label="Clustering Method",
                info="Instantly switch how familiar faces are grouped."
            )
            
    stats_panel = gr.HTML()
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("📊 Cluster List"):
                    summary_table = gr.Dataframe(
                        headers=["Cluster ID", "Images Count", "Mean Ridge Score"],
                        interactive=False,
                        max_height=600,
                        wrap=True
                    )
                with gr.TabItem("🔎 Global Image Search"):
                    gr.Markdown("Upload a picture to instantly check if it appears *anywhere* in the 43,000+ logs.")
                    img_upload = gr.Image(type="filepath", label="Upload profile image to search")
                    search_btn = gr.Button("Search Entire Log", variant="primary")
                    
        with gr.Column(scale=7):
            gr.Markdown("### 🔍 Cluster Details")
            selection_badge = gr.HTML(
                """
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 16px; text-align: center; color: #94a3b8; font-family: 'Inter', sans-serif;">
                    💡 Select a cluster from the list, or search by image to view results here.
                </div>
                """
            )
            
            gallery = gr.Gallery(
                label="Familiar Faces",
                show_label=False,
                elem_id="gallery",
                columns=[2, 3, 4],
                rows=[2],
                object_fit="contain",
                height=420,
                preview=True
            )
            
            gr.Markdown("#### 📋 Data Records")
            details_table = gr.Dataframe(
                interactive=False,
                max_height=220,
                wrap=True
            )

    def load_initial_data():
        df_total, df_familiar = load_and_process_data("Multi-Model Score Vector")
        summary = get_cluster_summary(df_familiar)
        stats_html = make_stats_html(df_total, df_familiar)
        return df_total, df_familiar, summary, summary, stats_html

    app.load(
        load_initial_data,
        inputs=None,
        outputs=[df_total_state, df_familiar_state, summary_state, summary_table, stats_panel]
    )

    method_radio.change(
        on_method_change,
        inputs=[method_radio],
        outputs=[df_total_state, df_familiar_state, summary_state, summary_table, stats_panel, gallery, details_table, selection_badge]
    )

    summary_table.select(
        on_cluster_select,
        inputs=[summary_state, df_familiar_state],
        outputs=[gallery, details_table, selection_badge]
    )
    
    search_btn.click(
        find_similar_image,
        inputs=[img_upload, df_familiar_state, summary_state, df_total_state],
        outputs=[gallery, details_table, selection_badge]
    )

if __name__ == "__main__":
    app.launch(theme=theme, css=custom_css, allowed_paths=[r"D:\BumbleLog"])
