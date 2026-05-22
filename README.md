# BumbleClaw Visual Preference Autoswipe

BumbleClaw is a local Bumble autoswipe system driven by your visual preferences. It reads the visible profile image from Bumble Web or a connected Android phone, scores it with models trained from your labels and historical swipe logs, then decides whether to swipe left or right.

This project started as a face-similarity rater. That baseline made the data loop concrete: label examples, embed them with InsightFace, compare against nearby references, and produce a score. The current system builds on that baseline:

1. InsightFace still provides face embeddings and a nearest-reference baseline.
2. Supervised regressors learn your `0-100` visual ratings from face embeddings.
3. CLIP image embeddings add profile-image context that face embeddings miss.
4. A binary preference layer learns `P(like)` from the score components that actually drive swipe decisions.
5. Live logs feed dynamic thresholds and new labeling rounds so each generation can be benchmarked against held-out Bumble screenshots.

The output is personal preference automation.

## Current Method

For the current personal runtime, the preferred branch is the Round3 fine-tuned MultimodalX line:

```text
research lineage = Multimodal X3 P85 and Multimodal X4 P80
repo presets     = multimodalx4 and multimodalx5
base stack       = bumble_combined_round3 rating system
decision layer   = Round3 score components + learned preference veto layer
runtime target   = dynamic P(like) threshold calibrated to recent logs
```

After recreating the matching private artifacts, run the P85 path with:

```powershell
python bumble_auto.py --setup multimodalx4 --log-dir <path-to-private-log-dir>
```

Run the P80 path with:

```powershell
python bumble_auto.py --setup multimodalx5 --log-dir <path-to-private-log-dir>
```

or on Android:

```powershell
python bumble_phone_auto.py --setup multimodalx4 --log-dir <path-to-private-log-dir>
python bumble_phone_auto.py --setup multimodalx5 --log-dir <path-to-private-log-dir>
```

These presets fine-tune the Round3 visual stack around actual swipe behavior:

- The base rating components come from the Round3 store, face regressor, multimodal regressor, and KNN reference score.
- A blend preference model estimates the older `P(like)` signal from the original score-component feature space.
- A second spline-logistic preference or veto layer is trained on disagreement-heavy Bumble decisions so it can correct profiles where the rating stack and the observed swipe preference diverge.
- Dynamic preference percentiles convert that learned probability into a live right-swipe boundary: P85 is stricter and P80 is less strict.
- The formula search keeps the score mathematically small: weighted ridge, multimodal, KNN, and learned probability components are tuned and then checked on held-out validation labels.

The P85 runnable preset is `multimodalx4`:

```text
Round3 artifacts = reference_store_bumble_combined_round3
formula          = MultimodalX2 score
weights          = 0.12 ridge + 0.05 multimodal
                   + 0.30 KNN + 0.53 old P(like)
veto model       = bumble_preference_multimodalx4
blend model      = bumble_preference_classifier
runtime          = dynamic preference P85, k=9
```

The P80 runnable preset is `multimodalx5`:

```text
Round3 artifacts = reference_store_bumble_combined_round3
formula          = tuned MultimodalX score
weights          = 0.60 ridge + 0.05 multimodal
                   + 0.20 KNN + 0.15 old P(like)
veto model       = bumble_preference_multimodalx5
blend model      = bumble_preference_classifier
runtime          = dynamic preference P80, k=11
```

The P85 and P80 finalist reports produced during my local experiments are:

- `results/multimodalx4_ratio_finalists_p85.csv`
- `results/multimodalx5_ratio_finalists_p80.csv`

The result rows use `x3__...` formula IDs because this branch was tuned from the MultimodalX3 Round3 veto/formula lane before the deployable presets were named `multimodalx4` and `multimodalx5`.

For comparison, the strongest earlier top-model benchmark in this repo is the `experimental1` stack:

```text
store       = bumble_combined_round2
score       = 0.30 * face_regressor + 0.70 * multimodal_regressor
decision    = spline-logistic binary preference model over score components
runtime     = dynamic P(like) threshold from recent logs
```

Run that earlier recommendation with:

```powershell
python bumble_auto.py --setup experimental1 --log-dir <path-to-private-log-dir>
```

or on Android:

```powershell
python bumble_phone_auto.py --setup experimental1 --log-dir <path-to-private-log-dir>
```

The preset is defined in `face_similarity/experimental_setup.py`. It selects:

- `embeddings/reference_store_bumble_combined_round2.npz`
- `models/rating_regressor_bumble_combined_round2.joblib`
- `models/rating_regressor_multimodal_bumble_combined_round2.joblib`
- `models/bumble_preference_experimental1.joblib`

### Why This Method

The model evolved by measuring each added layer.

| Stage | Method | What It Tests |
| --- | --- | --- |
| Face reference baseline | Weighted nearest labeled faces | Whether labels and embeddings contain a usable preference signal |
| Face regressor | Ridge, Random Forest, HistGradientBoosting | Whether supervised face embeddings beat nearest-reference averaging |
| Multimodal regressor | Face, CLIP, and face+CLIP Ridge/logistic-expected/MLP | Whether profile-image semantics improve rating prediction |
| Score blend | Face-biased mix of face and multimodal regressors | Whether a simple blend is more stable on Bumble screenshots |
| Preference layer | Thresholds, bucket rate, spline logistic, gradient boosting | Whether `P(like)` improves swipe decisions over raw score |
| Dynamic runtime | Recent score or `P(like)` percentiles | Whether the right-swipe rate stays calibrated as the profile stream shifts |

The benchmark workflow that led to these choices produces reports such as:

- `results/preference_top_model_best.csv` for the top scoring-generation recommendation.
- `results/bumble_preference_benchmark.csv` for score-threshold and preference-layer comparisons.
- `results/multimodal_regressor_eval*.csv` for rating MAE/RMSE across face, CLIP, and combined regressors.
- `results/benchmark_round3_locked_validation.csv` for a locked Bumble validation comparison between score stacks.

These reports, fitted models, embedding stores, screenshots, and label CSVs belong to the private experiment workspace. The public repo carries the scripts needed to regenerate them from user-procured data.

The earlier top-model benchmark decision path is:

```text
face_score       = face embedding regressor
multimodal_score = CLIP + face regressor
score            = 0.30 * face_score + 0.70 * multimodal_score
features         = score, face_score, multimodal_score, KNN score,
                   score spread, threshold distance, score bucket,
                   round flags, and related score metadata
p_like           = spline_logistic(features)
decision         = right if p_like >= preference_threshold else left
```

With the preference presets, the threshold is seeded from the trained model and can be recalibrated from the latest matching log history.

## Install

The code is Windows-oriented. Keep datasets, logs, model caches, and training workspaces outside the public repo when they contain personal images or swipe history.

Create the base Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
```

InsightFace downloads its model files on first use.

CLIP requires PyTorch and Transformers. CLIP-dependent methods can use a separate environment when the PyTorch install and model cache should live outside the base environment. Configure `CLIP_VENV_PYTHON` and `HF_CACHE_DIR` in `face_similarity/clip_runtime.py` for your machine, then create that environment if you use `face_biased`, `multimodal`, or MultimodalX methods:

```powershell
$CLIP_VENV = "<path-to-clip-venv>"
python -m venv $CLIP_VENV
& "$CLIP_VENV\Scripts\Activate.ps1"
cd <path-to-bumbleclaw>
pip install -r requirements.txt
pip install -r requirements-clip.txt
python -m playwright install chromium
```

Keep the Hugging Face and Torch caches outside the repo if they are large or shared across projects.

Check ONNX Runtime GPU support:

```powershell
python check_gpu.py
```

`CUDAExecutionProvider` is preferred. CPU mode still works, but live CLIP-assisted automation will be slower.

## Run Autoswipe

### Bumble Web

Start with one visible profile before enabling a loop:

```powershell
$LOG_DIR = "<path-to-private-log-dir>"
python bumble_auto.py --setup multimodalx4 --log-dir $LOG_DIR
```

Use `--setup multimodalx5` for the P80 variant. The first run opens a persistent Playwright profile in `.bumble_browser`. Log in to Bumble in that browser session if needed, leave a profile visible, and run again.

Run continuously with an explicit delay:

```powershell
python bumble_auto.py --setup multimodalx4 --log-dir $LOG_DIR --loop --delay 4
```

Run the long-loop mode with randomized delays and hourly pauses:

```powershell
python bumble_auto.py --setup multimodalx4 --log-dir $LOG_DIR --loop --247
```

Useful alternatives:

```powershell
# Current default Round3 score stack with a fixed score threshold
python bumble_auto.py --log-dir $LOG_DIR

# Preferred Round3 fine-tuned P85 and P80 paths
python bumble_auto.py --setup multimodalx4 --log-dir $LOG_DIR
python bumble_auto.py --setup multimodalx5 --log-dir $LOG_DIR

# Explicit preference decision layer
python bumble_auto.py --decision-mode preference --method face_biased --log-dir $LOG_DIR

# Score-only dynamic threshold
python bumble_auto.py --method face_biased --dynamic-from-logs --dynamic-percentile 80 --log-dir $LOG_DIR
```

The Web runner saves a current screenshot under `results/` and writes compressed profile logs to its configured log directory. Override that path with `--log-dir` when the logs should live outside the repo.

### Android Phone

Enable Android USB debugging, connect the phone, and make sure `adb devices` sees it.

Score and swipe the current Bumble screen once:

```powershell
python bumble_phone_auto.py --setup multimodalx4 --log-dir $LOG_DIR
```

Run in a loop:

```powershell
python bumble_phone_auto.py --setup multimodalx4 --log-dir $LOG_DIR --loop --delay 4
```

Use custom Android swipe coordinates for a different screen geometry:

```powershell
python bumble_phone_auto.py --setup multimodalx4 --log-dir $LOG_DIR `
  --left-swipe 850,1400,150,1400,250 `
  --right-swipe 150,1400,850,1400,250
```

Use `--setup multimodalx5` for the P80 variant. If multiple Android devices are attached, pass `--serial`.

## Runtime Choices

Named setups keep a benchmarked artifact bundle together after a user has generated the matching local artifacts:

| Setup | Main Idea |
| --- | --- |
| `experimental1` | Round2 face-biased score with the best top-model spline preference layer |
| `experimental2` | Round3 face-biased comparison |
| `multimodalx` | Formula score using ridge, multimodal, and `P(like)` with threshold decision |
| `multimodalx2` | Formula score that also uses KNN |
| `multimodalx3` | Round3 face-biased score with veto-style preference layer |
| `multimodalx4` | Preferred Round3 P85 path: MultimodalX2 score with blend and veto preference layers |
| `multimodalx5` | Preferred Round3 P80 path: tuned formula score with blend and veto preference layers |
| `multimodalx6` | Round2 veto comparison |

Use `--store`, `--regressor`, `--multimodal-regressor`, `--method`, `--face-weight`, `--preference-model`, and threshold flags when deliberately overriding a preset.

Available core scoring methods are:

- `knn`: weighted nearest-reference baseline.
- `regressor`: face-only supervised regressor.
- `multimodal`: learned multimodal regressor.
- `face_biased`: weighted blend of face-only and multimodal regressors.
- `multimodalx`, `multimodalx2`, `multimodalx5`: hand-tuned formula scores over score components and `P(like)`.

## Build The Visual Preference Data

### Dataset Specification

Users procure their own training images and logs, check that their use of those materials is lawful and appropriate, and label the data for their own preferences.

The rating pipeline expects a user-procured image set with:

- Image files reachable from CSV paths.
- A visible face for the InsightFace path when building face stores.
- Low, neutral, and high preference examples across the full rating range.
- Variation in crop, lighting, expression, pose, and profile-image context if the model should generalize beyond narrow portrait similarity.
- A held-out validation slice reserved for evaluation.

The base label CSV format is:

```csv
path,rating_1_5,rating
data/reference_images/example_001.jpg,5,100
data/reference_images/example_002.jpg,3,50
data/reference_images/example_003.jpg,1,0
```

`rating_1_5` is the human label. `rating` is its `0-100` numeric value:

```text
1 = 0
2 = 25
3 = 50
4 = 75
5 = 100
```

For a practical first pass, collect at least hundreds of labeled images with representation across the full rating scale. Bumble-specific calibration rounds need additional user-owned logged screenshots and labels from real decisions because live profile streams include screenshot framing and decision context beyond the base portrait labels.

### Label Base Reference Images

Run the labeling app over your local source image folders:

```powershell
python label_app.py --source-dir <path-to-reference-images> --output-csv <path-to-rating-labels.csv>
```

Keep high, low, and neutral examples so the model sees the whole preference range.

Optional dataset maintenance:

```powershell
python gender_cleanup.py --source-dir <path-to-reference-images> --provider cuda --det-thresh 0.25
python label_audit.py --min-gap 50 --min-similarity 0.9
```

### Build Rating Stores

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

Review failed face detections when needed:

```powershell
python export_rejected.py --labels $RATING_LABELS --provider cuda --det-thresh 0.25
```

## Train Rating Models

Train face-only rating regressors:

```powershell
python train_regressor.py `
  --store embeddings\reference_store.npz `
  --output models\rating_regressor.joblib `
  --report results\regressor_eval.csv
```

Train face, CLIP, and face+CLIP regressors:

```powershell
python train_multimodal_regressor.py `
  --face-store embeddings\reference_store.npz `
  --clip-store embeddings\clip_store.npz `
  --output models\rating_regressor_multimodal.joblib `
  --report results\multimodal_regressor_eval.csv
```

The multimodal report includes random and leak-aware validation. Prefer leak-aware comparisons when similar faces or repeated identities may exist across splits.

Score a folder manually before wiring a new model into automation:

```powershell
python score.py .\test_images `
  --method face_biased `
  --store embeddings\reference_store.npz `
  --regressor models\rating_regressor.joblib `
  --multimodal-regressor models\rating_regressor_multimodal.joblib `
  --csv results\scores.csv
```

## Train From Bumble Logs

Live automation writes screenshots and `scores.csv` into the configured log directory. Keep that directory private. Those logs are the feedback loop for later Bumble-specific rounds.

The Bumble-specific workflow expects user-owned logs with:

- Captured profile screenshots from the automation runner.
- A `scores.csv` written by the same run so score components, methods, thresholds, and screenshot paths can be joined later.
- Fresh human labels for selected screenshots, either rating labels for regressor rounds or binary `like` labels for preference rounds.
- A locked validation set selected before training new artifacts.

### Rating Round From Logged Screenshots

Prepare logged Bumble screenshots for rating labels:

```powershell
$RATING_ROUND = "<path-to-private-rating-round-workspace>"
python bumble_train.py prepare --source $LOG_DIR --output $RATING_ROUND
```

Label the selected screenshots:

```powershell
python label_app.py `
  --source-dir "$RATING_ROUND\selected" `
  --output-csv "$RATING_ROUND\labels\bumble_labels.csv" `
  --port 7863
```

Combine them with the base labels:

```powershell
python bumble_train.py combine-labels `
  --base $RATING_LABELS `
  --bumble-labels "$RATING_ROUND\labels\bumble_labels.csv" `
  --manifest "$RATING_ROUND\manifests\selection.csv" `
  --output "$RATING_ROUND\labels\combined_train_labels.csv"
```

Build new stores and train new regressors from that combined label file:

```powershell
python build_references.py `
  --labels "$RATING_ROUND\labels\combined_train_labels.csv" `
  --output embeddings\reference_store_bumble_combined.npz `
  --provider cuda `
  --det-thresh 0.25

python build_clip_store.py `
  --labels "$RATING_ROUND\labels\combined_train_labels.csv" `
  --store embeddings\reference_store_bumble_combined.npz `
  --output embeddings\clip_store_bumble_combined.npz `
  --provider cuda
```

Then run `train_regressor.py` and `train_multimodal_regressor.py` against those new stores with explicit output/report paths.

Evaluate locked Bumble labels against logged or rescored predictions:

```powershell
python bumble_train.py evaluate `
  --labels "$RATING_ROUND\labels\bumble_labels.csv" `
  --manifest "$RATING_ROUND\manifests\selection.csv" `
  --split validation `
  --output "$RATING_ROUND\reports\evaluation.csv"
```

### Binary Preference Layer

Prepare historical screenshots for binary preference labels:

```powershell
$PREFERENCE_ROUND = "<path-to-private-preference-round-workspace>"
python bumble_preference.py prepare --source $LOG_DIR --output $PREFERENCE_ROUND
```

Label binary preference decisions with the command printed by the prepare step, then train:

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

That benchmark is the path that produced the current `experimental1` recommendation.

## Inspect Runs

Open the original Gradio scoring UI:

```powershell
python app.py
```

Run the Next.js log dashboard:

```powershell
cd dashboard
npm install
npm run dev
```

Use the local URL printed by Next.js. The dashboard reads Bumble log history and gallery data from the log files used by automation.

## Repository Map

| Path | Purpose |
| --- | --- |
| `bumble_auto.py` | Bumble Web screenshot, score, decision, and keyboard automation |
| `bumble_phone_auto.py` | Android screenshot, score, decision, and ADB swipe automation |
| `face_similarity/` | Embeddings, scoring, regressors, preference features, logging, thresholds |
| `train_regressor.py` | Face-only rating model training |
| `train_multimodal_regressor.py` | Face/CLIP/multimodal rating model training |
| `bumble_train.py` | Bumble screenshot selection, combined labels, rating evaluation |
| `bumble_preference.py` | Binary preference data, benchmarks, and veto experiments |
| `models/` | Saved sklearn artifacts |
| `embeddings/` | InsightFace and CLIP stores |
| `results/` | Benchmarks, validation predictions, reports, and screenshots |
| `dashboard/` | Local Next.js log dashboard |

## Notes

- Keep model generation, store, regressor, multimodal regressor, and preference model together. Named setups exist to prevent mismatched artifacts.
- Prefer held-out Bumble screenshot benchmarks over intuition when choosing the next runtime stack.
- Rebuild stores after label, crop, embedding backend, or face-detection threshold changes.
- Use `--provider cuda` when you want GPU failures to surface during a run.
