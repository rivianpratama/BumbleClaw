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

Use the separate labeling app for the dataset folders at `references/Female Faces` and `references/women`:

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

The scorer uses the nearest 11 reference faces by default. Change that with `--k`.

## Local UI

```powershell
python app.py
```

Open the local Gradio URL printed in the terminal, usually `http://127.0.0.1:7860`.

## Notes

- Use clear reference images when possible.
- Add a range of ratings, not only images you rate highly.
- If the current store was built with DeepFace, delete or rebuild `embeddings/reference_store.npz`.
