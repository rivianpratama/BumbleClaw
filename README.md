# BumbleClaw Face Similarity Rating

Local face similarity scoring against your own labeled reference set. The tool uses pretrained InsightFace embeddings through ONNX Runtime. It does not train a deep model.

The `0-100` output means similarity to your labeled examples. It is not an objective attractiveness score.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

InsightFace downloads model files on first use.

## Check NVIDIA GPU

```powershell
python check_gpu.py
```

You want to see:

```text
CUDAExecutionProvider
```

If CUDA is not listed, the code can still run on CPU. GPU mode depends on ONNX Runtime GPU being able to find compatible NVIDIA CUDA/cuDNN runtime libraries on Windows.

## Label References

Create `labels.csv` in the repo root:

```csv
path,rating
references/1.jpg,95
references/2.jpg,70
references/3.jpg,20
```

Use ratings from `0` to `100`. Include both high and low examples so the rating has a useful range.

## Label A Large Dataset

Use the separate labeling app for the configured dataset folders:

- `references/Female Faces`
- `references/women`
- `references/archive (4)/Data_all`
- `references/Selfie`
- `references/Selfie-version2`
- `references/Kpop_Profile_curated`

```powershell
python label_app.py
```

For phone labeling, keep your computer and iPhone on the same Wi-Fi. The app runs on `0.0.0.0:7861` and prints one or more local URLs like:

```text
http://192.168.1.23:7861
```

Open that URL in iPhone Safari.

The app writes `dataset_labels.csv` after every rating:

```csv
path,rating_1_5,rating
references/Female Faces/0 (1).jpeg,5,100
references/Female Faces/0 (2).jpg,3,50
references/Female Faces/0 (3).png,1,0
```

Rating scale:

```text
1 = 0   extremely dislike
2 = 25  dislike
3 = 50  neutral
4 = 75  like
5 = 100 extremely like
```

The app resumes at the next unlabeled image when reopened.

## Clean Selfie Gender

Quarantine male-predicted selfie images before labeling:

```powershell
python gender_cleanup.py --source-dir .\references\Selfie --provider cuda --det-thresh 0.25
```

This moves male predictions into `references/Selfie_removed_men` and writes `results/selfie_gender_cleanup.csv`. Use `--dry-run` first if you only want the report.

## Curate K-pop Profile Candidates

Copy a balanced manual-label subset from the Bumble-like K-pop dataset:

```powershell
python kpop_profile_curate.py --source-dir "D:\KPOP dataset" --output-dir references\Kpop_Profile_curated --report results\kpop_profile_curate.csv --provider cuda --det-thresh 0.25 --target-count 400 --max-per-identity 14
```

This writes `results/kpop_profile_curate.csv` and copies selected images into `references/Kpop_Profile_curated`.

## Build Embeddings

```powershell
python build_references.py --labels labels.csv
python build_references.py --labels dataset_labels.csv --provider cuda
python build_references.py --labels dataset_labels.csv --provider cuda --det-thresh 0.25
```

This creates `embeddings/reference_store.npz`. Rebuild it whenever you change reference images, ratings, or the embedding backend.

To copy images that fail face detection into a review folder:

```powershell
python export_rejected.py --labels dataset_labels.csv --provider cuda
python export_rejected.py --labels dataset_labels.csv --provider cuda --det-thresh 0.25
```

This writes `results/rejected_faces.csv` and copies files into `results/rejected_faces`.

Provider options:

```powershell
python build_references.py --labels labels.csv --provider auto
python build_references.py --labels labels.csv --provider cuda
python build_references.py --labels labels.csv --provider cpu
```

Use `--provider cuda` when you want the command to fail instead of falling back to CPU.

## Score Images

```powershell
python score.py .\test_images
python score.py .\test_images\photo.jpg
python score.py .\test_images --csv results\scores.csv
```

The scorer uses the nearest 20 reference faces by default. Change that with `--k`.

## Train A Rating Regressor

After building `embeddings/reference_store.npz`, train a supervised model from the face embeddings:

```powershell
python train_regressor.py
```

This compares Ridge, RandomForest, and HistGradientBoosting regressors, saves the best model to `models/rating_regressor.joblib`, and writes metrics to `results/regressor_eval.csv`.

Use the trained regressor from the command line:

```powershell
python score.py .\test_images --method regressor
python score.py .\test_images\photo.jpg --method regressor
```

The local Gradio UI automatically uses `models/rating_regressor.joblib` when it exists, while still showing nearest reference faces for debugging.

## Test CLIP + Face Multimodal Scoring

CLIP/PyTorch is large. If your repo and default venv are on `C:`, create a separate venv on a drive with enough free space, for example:

```powershell
D:
python -m venv D:\BumbleClawClipVenv
D:\BumbleClawClipVenv\Scripts\activate
cd C:\Users\Rivian\Documents\GitHub\BumbleClaw
pip install -r requirements.txt
pip install -r requirements-clip.txt
```

Build CLIP embeddings aligned to the current face store:

```powershell
python build_clip_store.py --labels dataset_labels.csv --store embeddings\reference_store.npz --output embeddings\clip_store.npz --provider cuda
```

Train and compare face-only, CLIP-only, and combined models:

```powershell
python train_multimodal_regressor.py `
  --face-store embeddings\reference_store.npz `
  --clip-store embeddings\clip_store.npz `
  --output models\rating_regressor_multimodal.joblib `
  --report results\multimodal_regressor_eval.csv
```

The report includes both random validation and leak-aware validation. Keep the original Ridge model as the trusted default unless the multimodal model beats it on leak-aware MAE/RMSE.

For a small smoke test:

```powershell
python build_clip_store.py --labels dataset_labels.csv --store embeddings\reference_store.npz --output embeddings\clip_store_smoke.npz --provider auto --limit 5
```

## Audit And Normalize Label Conflicts

Find very similar faces with conflicting labels:

```powershell
python label_audit.py
python label_audit.py --min-gap 50 --min-similarity 0.9
```

Create a separate model-building CSV that averages labels within high-similarity face groups:

```powershell
python normalize_labels.py --similarity-threshold 0.9
python build_references.py --labels dataset_labels_normalized.csv --output embeddings\reference_store_normalized.npz --provider cuda --det-thresh 0.25
python train_regressor.py --store embeddings\reference_store_normalized.npz --output models\rating_regressor_normalized.joblib --report results\regressor_eval_normalized.csv
```

This does not modify `dataset_labels.csv`.

## Local UI

```powershell
python app.py
```

Open the local Gradio URL printed in the terminal, usually `http://127.0.0.1:7860`.

## Notes

- Use clear reference images when possible.
- Add a range of ratings, not only images you rate highly.
- If the current store was built with DeepFace, delete or rebuild `embeddings/reference_store.npz`.
