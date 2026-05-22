<div align="center">
  <h1>BumbleClaw</h1>
  <p><strong>A local visual-preference research loop for Bumble screenshots.</strong></p>
  <p>
    Capture -> label -> embed -> train -> benchmark -> relabel the hard cases -> calibrate live decisions.
  </p>
  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-ML%20pipeline-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img alt="Local first" src="https://img.shields.io/badge/Data-local%20first-1F6FEB?style=for-the-badge">
    <img alt="InsightFace and CLIP" src="https://img.shields.io/badge/Vision-InsightFace%20%2B%20CLIP-0F766E?style=for-the-badge">
    <img alt="Dashboard" src="https://img.shields.io/badge/Dashboard-Next.js-111111?style=for-the-badge&logo=nextdotjs&logoColor=white">
  </p>
</div>

> [!IMPORTANT]
> The public workflow should contain the process and tools, not private screenshots, labels, embeddings, fitted models, browser state, or swipe logs from one person's trained behavior. Review tracked artifacts before publishing a clone or release.

BumbleClaw can capture the visible profile from Bumble Web or a connected Android phone, score the image with locally trained models, log the score components, and optionally turn that output into a left or right swipe. The repository keeps the experiment history visible because the failures changed the design.

It is not a public preference dataset and it is not a portable recommendation model. A model trained from one person's labels is not a general ranking of other people.

Users are responsible for the legality, platform rules, privacy handling, and appropriateness of any data they collect or automate.

<p align="center">
  <a href="#the-journey">Journey</a> |
  <a href="#pipeline-overview">Pipeline</a> |
  <a href="#replicate-the-process">Replication</a> |
  <a href="#runtime-options">Runtime</a> |
  <a href="#dashboard-and-inspection">Dashboard</a>
</p>

## At A Glance

| Surface | What It Does |
| --- | --- |
| Capture | Reads Bumble Web screenshots or Android screen captures. |
| Rating | Scores image evidence with KNN, face regressors, multimodal regressors, and blend formulas. |
| Decision | Compares score thresholds or learned preference probabilities against static or dynamic boundaries. |
| Feedback | Selects logged screenshots for fresh labels, screenshot rounds, preference labels, and disagreement checks. |
| Inspection | Keeps score components, thresholds, screenshots, and setup metadata visible through local reports and a dashboard. |

## The Journey

This repo did not begin as a grand autoswipe system. It started as a face-similarity rater and got more complicated only when the next benchmark or live log exposed a specific gap.

```text
face similarity
-> 1-5 rating labels
-> supervised face score
-> CLIP-aware screenshot score
-> face-biased blends
-> dynamic score thresholds
-> binary P(like) layer
-> disagreement-only veto labels
-> named experiment setups
```

| Phase | What Changed | Why It Changed |
| --- | --- | --- |
| Face reference baseline | KNN over labeled InsightFace embeddings | The first goal was to prove that labels could drive a repeatable score at all. |
| Rating models | Ridge and other supervised regressors over face embeddings | Nearest examples were interpretable but too brittle to be the whole scorer. |
| Multimodal rating | Face + CLIP image signals | Bumble screenshots are not clean portraits: crop, pose, lighting, app framing, and visible context matter. |
| Bumble-specific rounds | Re-label logged screenshots and combine them with earlier labels | The screenshot domain did not behave like the original reference image pool. |
| Dynamic thresholds | Use recent score percentiles instead of trusting one fixed cutoff forever | The shown profile stream drifts, and a threshold that felt right in one slice can become too strict or too loose later. |
| Binary preference labels | Train a `P(like)` decision layer from actual left/right intent | A five-point attractiveness/rating label and a real swipe decision are related but not the same target. |
| Veto / disagreement evaluation | Label profiles where model decisions disagree | Easy unanimous examples inflate confidence; hard boundary cases expose the decision layer. |
| Named setups | Keep Round2, Round3, MultimodalX, and veto experiments explicit | Artifact bundles, formula weights, KNN `k`, and threshold policy can be valid alone but wrong together. |

The experiment stayed honest about dead ends:

- Fixed thresholds were easier to explain, but runtime score distributions shifted.
- More formula components did not automatically reduce swipe mistakes.
- A learned preference layer could improve a score stack and still feel wrong live if its calibration or training labels were mismatched.
- A multi-model vote sounded attractive, but repeated scoring and repeated history scans have real runtime cost. It should earn its complexity in a benchmark.

The public README intentionally avoids publishing private labels and local validation rows. It keeps the process, the experiment lineage, and the concerns that made each next round necessary.

## Concerns That Shape The Design

### Private data

Profile screenshots, swipe logs, label CSVs, browser state, embedding stores, and fitted preference models can all expose personal information. Keep them outside a public repository. The `.gitignore` is intentionally aggressive around generated images, CSVs, embeddings, models, and logs.

### Rating versus decision

The rating pipeline answers a question like "what numeric visual score would this labeler give this image?" The preference pipeline answers a different question: "given the current score components and prior labels, would this labeler choose right?" The second layer exists because a single `0-100` threshold often misses non-monotonic or context-dependent decisions.

### Screenshot-domain mismatch

Clean reference photos and Bumble screenshots are not the same input distribution. Screenshot rounds exist to test crops, app chrome, visible text, framing, low-quality logs, and profiles near the decision boundary. Keep locked validation examples before training new screenshot rounds.

### Calibration drift

The shown pool can shift during a run. Runtime supports static thresholds and recent-history percentile thresholds over either score values or preference probabilities. Dynamic thresholds are calibration tools; they do not make a weak model accurate by themselves.

### Sensitive attributes

Filtering rules that the app exposes to the user belong in the app settings or explicit profile data path. The visual model should not infer sensitive attributes from appearance as a shortcut.

### Artifact mismatch

A reference store, face regressor, multimodal regressor, preference model, formula score, KNN `k`, crop preset, and threshold policy can be individually valid but wrong together. Named setups keep experimental bundles together after the matching local artifacts have been regenerated.

## Pipeline Overview

The main score components are:

- **KNN**: weighted nearest labeled face embeddings.
- **Face regressor**: a supervised rating model over face embeddings.
- **Multimodal regressor**: a supervised rating model that can use face and CLIP image features.
- **Face-biased score**: a weighted blend of face and multimodal ratings.
- **Preference probability**: a classifier over score components and derived features such as component spread, score bucket, and distance from threshold.

```mermaid
flowchart LR
    A[Visible profile screenshot] --> B[Face embedding]
    A --> C[CLIP image embedding]
    B --> D[KNN reference score]
    B --> E[Face regressor]
    B --> F[Multimodal regressor]
    C --> F
    D --> G[Score components]
    E --> G
    F --> G
    G --> H[Raw score or formula blend]
    G --> I[Preference features]
    H --> J[Static or dynamic score threshold]
    I --> K[P like or veto layer]
    K --> L[Static or dynamic probability threshold]
    J --> M[Left or right decision]
    L --> M
    M --> N[Logs, dashboard, relabel queue]
```

For later experiments the "final score" shown in logs may be a calibrated preference probability scaled to `0-100`, while component scores remain available for debugging and benchmarks.

## Install

The project is Windows-oriented.

Create the base Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
```

InsightFace downloads its local model files on first use.

CLIP-dependent methods require PyTorch and Transformers. The code supports a separate CLIP environment and cache so large model dependencies do not need to live in the public repo. Configure `CLIP_VENV_PYTHON` and `HF_CACHE_DIR` in `face_similarity/clip_runtime.py` for the machine that runs CLIP, then create that environment when needed:

```powershell
$CLIP_VENV = "<path-to-clip-venv>"
python -m venv $CLIP_VENV
& "$CLIP_VENV\Scripts\Activate.ps1"
cd <path-to-bumbleclaw>
pip install -r requirements.txt
pip install -r requirements-clip.txt
python -m playwright install chromium
```

Check ONNX Runtime GPU support:

```powershell
python check_gpu.py
```

`CUDAExecutionProvider` is preferred for live CLIP-assisted runs. CPU mode is useful for simpler checks but will be slower.

## Replicate The Process

### 1. Build a base rating dataset

Start with user-procured images that you are allowed to use. Keep low, neutral, and high examples so the score range is not trained only around one outcome.

The rating label CSV format is:

```csv
path,rating_1_5,rating
data/reference_images/example_001.jpg,5,100
data/reference_images/example_002.jpg,3,50
data/reference_images/example_003.jpg,1,0
```

The numeric mapping used by the rating tools is:

```text
1 = 0
2 = 25
3 = 50
4 = 75
5 = 100
```

Label local source images:

```powershell
python label_app.py --source-dir <path-to-reference-images> --output-csv <path-to-rating-labels.csv>
```

Optional label check:

```powershell
python label_audit.py --min-gap 50 --min-similarity 0.9
```

### 2. Build stores and rating models

Build the face reference store:

```powershell
$RATING_LABELS = "<path-to-rating-labels.csv>"
python build_references.py --labels $RATING_LABELS --provider cuda --det-thresh 0.25
```

Build the aligned CLIP store:

```powershell
python build_clip_store.py `
  --labels $RATING_LABELS `
  --store embeddings\reference_store.npz `
  --output embeddings\clip_store.npz `
  --provider cuda
```

Train face-only rating regressors:

```powershell
python train_regressor.py `
  --store embeddings\reference_store.npz `
  --output models\rating_regressor.joblib `
  --report results\regressor_eval.csv
```

Train face, CLIP, and combined regressors:

```powershell
python train_multimodal_regressor.py `
  --face-store embeddings\reference_store.npz `
  --clip-store embeddings\clip_store.npz `
  --output models\rating_regressor_multimodal.joblib `
  --report results\multimodal_regressor_eval.csv
```

Prefer leak-aware comparisons when repeated people or near-duplicate screenshots may exist across train and validation data.

### 3. Score before automating

Score a folder manually:

```powershell
python score.py .\test_images `
  --method face_biased `
  --store embeddings\reference_store.npz `
  --regressor models\rating_regressor.joblib `
  --multimodal-regressor models\rating_regressor_multimodal.joblib `
  --csv results\scores.csv
```

Check score distributions, prediction failures, and held-out error before connecting a new artifact set to automation.

### 4. Collect screenshot-domain logs

Run Bumble Web once on a visible profile:

```powershell
$LOG_DIR = "<path-to-private-log-dir>"
python bumble_auto.py --log-dir $LOG_DIR
```

The first web run uses a persistent Playwright browser state under `.bumble_browser`. Log in there when needed and keep that state private.

For Android, enable USB debugging and check that `adb devices` can see the phone:

```powershell
python bumble_phone_auto.py --log-dir $LOG_DIR
```

Looping is explicit:

```powershell
python bumble_auto.py --log-dir $LOG_DIR --loop --delay 4
python bumble_phone_auto.py --log-dir $LOG_DIR --loop --delay 4
```

Logs include screenshots and a `scores.csv` with component scores, thresholds, selected setup metadata, and model paths. Treat that directory as private training material.

### 5. Train screenshot rounds

Prepare a logged screenshot round for five-point rating labels:

```powershell
$RATING_ROUND = "<path-to-private-rating-round-workspace>"
python bumble_train.py prepare --source $LOG_DIR --output $RATING_ROUND
```

Label it:

```powershell
python label_app.py `
  --source-dir "$RATING_ROUND\selected" `
  --output-csv "$RATING_ROUND\labels\bumble_labels.csv" `
  --port 7863
```

Combine screenshot labels with the base rating labels:

```powershell
python bumble_train.py combine-labels `
  --base $RATING_LABELS `
  --bumble-labels "$RATING_ROUND\labels\bumble_labels.csv" `
  --manifest "$RATING_ROUND\manifests\selection.csv" `
  --output "$RATING_ROUND\labels\combined_train_labels.csv"
```

Rebuild stores and regressors from the combined label file. Use explicit artifact and report names for each round so old validation results remain comparable.

### 6. Train a binary preference layer

Prepare screenshot-domain binary labels:

```powershell
$PREFERENCE_ROUND = "<path-to-private-preference-round-workspace>"
python bumble_preference.py prepare --source $LOG_DIR --output $PREFERENCE_ROUND
```

Label the prepared images as binary left or right intent, then train:

```powershell
python bumble_preference.py train `
  --manifest "$PREFERENCE_ROUND\manifests\selection.csv" `
  --labels "$PREFERENCE_ROUND\labels\binary_preference_labels.csv" `
  --output models\bumble_preference_classifier.joblib `
  --report results\bumble_preference_benchmark.csv
```

Benchmark score generations with a preference layer on top:

```powershell
python bumble_preference.py benchmark-models `
  --manifest "$PREFERENCE_ROUND\manifests\selection.csv" `
  --labels "$PREFERENCE_ROUND\labels\binary_preference_labels.csv" `
  --preference-model models\bumble_preference_classifier.joblib `
  --output results\preference_top_model_benchmark.csv `
  --best-output results\preference_top_model_best.csv `
  --provider cuda
```

### 7. Evaluate disagreement cases

Later experiments can focus labeling on profiles where several decision paths disagree:

```powershell
python bumble_preference.py prepare-veto-eval --source $LOG_DIR --output <path-to-private-veto-workspace>
```

Use the labeling command printed by that step. Then compare the labeled veto decisions with model outputs:

```powershell
python bumble_preference.py report-veto-eval
python bumble_preference.py benchmark-veto-layers
```

This workflow is useful when easy examples dominate the logs but the real engineering question is which decision layer handles boundary cases better.

## Runtime Options

Named setups in `face_similarity/experimental_setup.py` are lab presets. They are conveniences for a matching local artifact bundle, not universal recommendations. Recreate or replace the referenced stores and fitted models before expecting them to work on another machine.

### Setup Lineage

The setup names preserve the research path instead of rewriting it after every winner changed.

| Setup | Role In The Journey |
| --- | --- |
| `experimental1` | Earlier Round2 face-biased stack with a binary preference layer. |
| `experimental2` | Round3 comparison for the same preference-layer idea. |
| `multimodalx` | Formula score that mixes rating components with an earlier `P(like)` signal. |
| `multimodalx2` | Formula score that adds KNN into that blend. |
| `multimodalx3` | Round3 score components with a disagreement-trained veto-style layer. |
| `multimodalx4` | Veto layer on top of the MultimodalX2-style branch. |
| `multimodalx5` | Another tuned formula branch tested against held-out veto labels. |
| `multimodalx6` | Round2 revisited with the newer veto-spline decision layer. |

That lineage matters because it records the question each setup was trying to answer. It also makes benchmark reports easier to audit when a simpler older branch beats a newer one on a specific validation slice.

Core scoring methods are:

- `knn`
- `regressor`
- `multimodal`
- `face_biased`
- formula methods such as `multimodalx`, `multimodalx2`, and `multimodalx5`

Use explicit flags when overriding a preset deliberately:

```text
--store
--regressor
--multimodal-regressor
--method
--face-weight
--preference-model
--threshold
--dynamic-from-logs
--dynamic-preference-from-logs
```

Dynamic threshold flags can operate on raw score history or on preference-probability history. Keep that distinction clear when interpreting logs.

## Dashboard And Inspection

Open the original Gradio scoring UI:

```powershell
python app.py
```

Run the local Next.js dashboard:

```powershell
cd dashboard
npm install
npm run dev
```

The dashboard reads local score history and screenshots from the automation log directory. Do not deploy a dashboard instance that exposes private logs without reviewing its access boundary.

## Repository Map

| Path | Purpose |
| --- | --- |
| `bumble_auto.py` | Bumble Web screenshot, score, decision, and keyboard automation |
| `bumble_phone_auto.py` | Android screenshot, score, decision, and ADB swipe automation |
| `face_similarity/` | Embeddings, scoring, regressors, preference features, logging, thresholds |
| `train_regressor.py` | Face-only rating model training |
| `train_multimodal_regressor.py` | Face/CLIP/multimodal rating model training |
| `bumble_train.py` | Screenshot selection, combined labels, and rating evaluation |
| `bumble_preference.py` | Binary preference data, benchmarks, and disagreement experiments |
| `label_app.py` | Five-point and binary local labeling UI |
| `dashboard/` | Local score-history dashboard |

## Practical Notes

- Keep private screenshots, labels, logs, models, and embeddings out of public commits.
- Hold out validation examples before tuning a new score formula or preference layer.
- Rebuild artifact bundles after crop, label, embedding backend, or face-detection changes.
- Treat dynamic percentile settings as a runtime calibration choice and benchmark them against real held-out decisions.
- Prefer a simpler model when it matches the more complicated model on validation and is easier to reason about live.
