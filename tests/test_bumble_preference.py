from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import bumble_preference
from face_similarity.labeling import LabelRow


class BumblePreferenceTests(unittest.TestCase):
    def test_model_grid_only_includes_available_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "embeddings").mkdir()
            (root / "models").mkdir()
            touch(root / "embeddings" / "reference_store.npz")
            touch(root / "models" / "rating_regressor.joblib")
            touch(root / "models" / "rating_regressor_multimodal.joblib")
            touch(root / "embeddings" / "reference_store_normalized.npz")
            touch(root / "models" / "rating_regressor_normalized.joblib")

            with patch("bumble_preference.can_load_regressor", return_value=True):
                configs = bumble_preference.model_grid(root)

        ids = {config.config_id for config in configs}
        self.assertIn("original__knn", ids)
        self.assertIn("original__ridge", ids)
        self.assertIn("original__multimodal", ids)
        self.assertIn("original__face_biased__w0p44", ids)
        self.assertIn("normalized__knn", ids)
        self.assertIn("normalized__ridge", ids)
        self.assertNotIn("normalized__multimodal", ids)

    def test_chronological_split_uses_oldest_train_newest_validation(self) -> None:
        rows = [{"timestamp": str(value)} for value in [5, 1, 4, 2, 3]]

        train_rows, validation_rows = bumble_preference.chronological_split(rows, train_fraction=0.6)

        self.assertEqual([row["timestamp"] for row in train_rows], ["1", "2", "3"])
        self.assertEqual([row["timestamp"] for row in validation_rows], ["4", "5"])

    def test_derive_config_score_rows_uses_face_weight(self) -> None:
        bundle = bumble_preference.ModelBundle(
            generation="test_round3",
            store_path=Path("store.npz"),
            regressor_path=Path("ridge.joblib"),
            multimodal_regressor_path=Path("multi.joblib"),
            default_face_weight=0.44,
            deploy_rank=0,
        )
        config = bumble_preference.ModelConfig(bundle=bundle, method="face_biased", face_weight=0.25)
        component_rows = [
            {
                "timestamp": "1",
                "selected_path": "a.jpg",
                "ridge": "20",
                "multimodal": "80",
                "knn": "50",
                "regressor_path": "ridge_round3.joblib",
                "multimodal_regressor_path": "multi_round3.joblib",
            }
        ]

        rows = bumble_preference.derive_config_score_rows(config, component_rows, threshold=55.0)

        self.assertEqual(rows[0]["score"], "65.000000")
        self.assertEqual(rows[0]["face_biased"], "65.000000")
        self.assertEqual(rows[0]["face_weight"], "0.250000")
        self.assertEqual(rows[0]["distance_from_threshold"], "10.000000")

    def test_probability_threshold_for_rate_targets_requested_count(self) -> None:
        probabilities = np.asarray([0.9, 0.7, 0.4, 0.2, 0.1], dtype=np.float32)

        threshold = bumble_preference.probability_threshold_for_rate(probabilities, 0.4)

        self.assertAlmostEqual(threshold, 0.7)
        self.assertEqual(int((probabilities >= threshold).sum()), 2)

    def test_write_benchmark_report_writes_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.csv"
            row = {field: "" for field in bumble_preference.MODEL_BENCHMARK_FIELDS}
            row.update({"strategy": "reuse_current", "generation": "original", "swipe_errors": 1})

            bumble_preference.write_rows(path, [row], bumble_preference.MODEL_BENCHMARK_FIELDS)

            with path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
        self.assertEqual(reader.fieldnames, bumble_preference.MODEL_BENCHMARK_FIELDS)
        self.assertEqual(rows[0]["strategy"], "reuse_current")

    def test_excluded_veto_training_stems_uses_stores_and_preference_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "store.npz"
            np.savez_compressed(store, paths=np.asarray([r"D:\BumbleLog\profile_store.webp"]))
            manifest = root / "selection.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["screenshot", "selected_path"])
                writer.writeheader()
                writer.writerow({"screenshot": "profile_manifest.jpg", "selected_path": ""})

            stems = bumble_preference.excluded_veto_training_stems([store], manifest)

        self.assertEqual(stems, {"profile_store", "profile_manifest"})

    def test_eligible_veto_candidates_excludes_stems_and_dedupes_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            touch(source / "profile_keep.jpg")
            touch(source / "profile_excluded.jpg")
            touch(source / "profile_duplicate.webp")
            rows = [
                {"timestamp": "3", "screenshot": "profile_duplicate.webp"},
                {"timestamp": "2", "screenshot": "profile_excluded.jpg"},
                {"timestamp": "1", "screenshot": "profile_keep.jpg"},
                {"timestamp": "4", "screenshot": "profile_duplicate.webp"},
            ]

            candidates = bumble_preference.eligible_veto_candidates(
                rows,
                source=source,
                excluded_stems={"profile_excluded"},
            )

        self.assertEqual([candidate.row["screenshot"] for candidate in candidates], ["profile_keep.jpg", "profile_duplicate.webp"])

    def test_simulate_dynamic_decisions_uses_prior_window_history(self) -> None:
        decisions = bumble_preference.simulate_dynamic_decisions(
            [10.0, 20.0, 30.0, 100.0],
            fixed_threshold=50.0,
            dynamic_window=2,
            dynamic_min_history=2,
            min_threshold=0.0,
            max_threshold=100.0,
        )

        self.assertEqual(decisions[:2], [(50.0, "left"), (50.0, "left")])
        self.assertAlmostEqual(decisions[2][0], 18.0)
        self.assertEqual(decisions[2][1], "right")
        self.assertAlmostEqual(decisions[3][0], 28.0)

    def test_apply_veto_dynamic_decisions_uses_score_and_probability_thresholds(self) -> None:
        rows = [
            {
                "round3_score": "60",
                "multimodalx_score": "60",
                "multimodalx2_score": "60",
                "experimental1_probability": "0.60",
            }
        ]

        bumble_preference.apply_veto_dynamic_decisions(rows, dynamic_window=200, dynamic_min_history=50)

        self.assertEqual(rows[0]["round3_action"], "right")
        self.assertEqual(rows[0]["experimental1_action"], "right")
        self.assertAlmostEqual(float(rows[0]["round3_threshold"]), 55.0)
        self.assertAlmostEqual(float(rows[0]["experimental1_threshold"]), 0.556059)

    def test_disagreement_veto_rows_excludes_unanimous_votes(self) -> None:
        rows = [
            {"round3_action": "left", "multimodalx_action": "left", "multimodalx2_action": "left", "experimental1_action": "left"},
            {"round3_action": "right", "multimodalx_action": "left", "multimodalx2_action": "right", "experimental1_action": "left"},
        ]

        selected = bumble_preference.disagreement_veto_rows(rows)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["disagreement_pattern"], "RLRL")

    def test_build_veto_report_rows_counts_model_agreement(self) -> None:
        manifest = [
            veto_manifest_row("a.jpg", round3="right", multimodalx="left", multimodalx2="right", experimental1="left"),
            veto_manifest_row("b.jpg", round3="right", multimodalx="right", multimodalx2="left", experimental1="left"),
        ]
        labels = [
            LabelRow(path="a.jpg", rating_1_5=5, rating=100),
            LabelRow(path="b.jpg", rating_1_5=1, rating=0),
        ]

        details, summary = bumble_preference.build_veto_report_rows(
            manifest,
            labels=labels,
            stats={"disagreement_selected_count": "2"},
        )

        round3 = next(row for row in summary if row["model"] == "Round3")
        self.assertEqual(len(details), 2)
        self.assertEqual(round3["agreement_count"], 1)
        self.assertEqual(round3["false_positive"], 1)
        self.assertEqual(round3["false_negative"], 0)
        self.assertEqual(round3["disagreement_selected_count"], "2")

    def test_write_veto_manifest_writes_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disagreement.csv"
            row = {field: "" for field in bumble_preference.VETO_MANIFEST_FIELDS}
            row.update({"timestamp": "1", "disagreement_pattern": "RLLR"})

            bumble_preference.write_rows(path, [row], bumble_preference.VETO_MANIFEST_FIELDS)

            with path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
        self.assertEqual(reader.fieldnames, bumble_preference.VETO_MANIFEST_FIELDS)
        self.assertEqual(rows[0]["disagreement_pattern"], "RLLR")

    def test_veto_layer_rows_build_original_round2_round3_and_multimodalx2_feature_rows(self) -> None:
        manifest = [
            {
                "timestamp": "1",
                "screenshot": "a.jpg",
                "selected_path": "selected/a.jpg",
                "like": "1",
                "experimental1_score": "52",
                "experimental1_threshold": "0.55",
                "experimental1_action": "right",
                "round3_score": "58",
                "round3_threshold": "55",
                "round3_action": "right",
                "multimodalx2_score": "62",
                "multimodalx2_threshold": "56",
                "multimodalx2_action": "left",
            }
        ]
        components = [
            {
                "timestamp": "1",
                "screenshot": "a.jpg",
                "ridge": "40",
                "multimodal": "80",
                "knn_k9": "35",
                "knn_k11": "70",
                "regressor_path": "ridge_round3.joblib",
                "multimodal_regressor_path": "multi_round3.joblib",
            }
        ]

        original = bumble_preference.veto_layer_rows(manifest, components, lane="Original")
        round2 = bumble_preference.veto_layer_rows(manifest, components, lane="Round2")
        round3 = bumble_preference.veto_layer_rows(manifest, components, lane="Round3")
        multimodalx2 = bumble_preference.veto_layer_rows(manifest, components, lane="MultimodalX2")

        self.assertEqual(original[0]["score"], "60.000000")
        self.assertEqual(original[0]["face_weight"], "0.500000")
        self.assertEqual(original[0]["action"], "right")
        self.assertEqual(original[0]["threshold"], "55.000000")
        self.assertEqual(round2[0]["score"], "52.000000")
        self.assertEqual(round2[0]["knn"], "70.000000")
        self.assertEqual(round2[0]["face_weight"], "0.300000")
        self.assertEqual(round2[0]["action"], "right")
        self.assertEqual(round2[0]["threshold"], "55.000000")
        self.assertEqual(round3[0]["score"], "58.000000")
        self.assertEqual(round3[0]["knn"], "70.000000")
        self.assertEqual(round3[0]["face_weight"], "0.440000")
        self.assertEqual(multimodalx2[0]["score"], "62.000000")
        self.assertEqual(multimodalx2[0]["knn"], "35.000000")
        self.assertEqual(multimodalx2[0]["action"], "left")

    def test_veto_layer_benchmark_row_writes_report_metrics(self) -> None:
        result = bumble_preference.metrics_from_predictions(
            "spline_logistic",
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0.2, 0.8], dtype=np.float32),
            0.5,
        )

        row = bumble_preference.veto_layer_benchmark_row("Round3", "retrained_layer", 0.25, result)

        self.assertEqual(row["lane"], "Round3")
        self.assertEqual(row["model"], "spline_logistic")
        self.assertEqual(row["swipe_errors"], 0)

    def test_veto_x3_formula_grid_includes_reference_formulas_and_variants(self) -> None:
        formulas = bumble_preference.veto_x3_formula_grid(0.5)

        ids = {formula.config_id for formula in formulas}
        self.assertIn("x3__k9__r0p44_m0p56_k0_p0__score_plike_only", ids)
        self.assertIn("x3__k11__r0p12_m0p05_k0p3_p0p53__with_plike_feature", ids)
        self.assertIn("x3__k9__r0p5_m0_k0p5_p0__with_plike_feature", ids)

    def test_veto_x3_formula_rows_blend_all_round3_signals(self) -> None:
        formula = bumble_preference.VetoX3Formula(
            ridge_weight=0.10,
            multimodal_weight=0.20,
            knn_weight=0.30,
            old_p_like_weight=0.40,
            knn_k=9,
            explicit_old_p_like=True,
        )
        manifest = [
            {
                "timestamp": "1",
                "screenshot": "a.jpg",
                "selected_path": "selected/a.jpg",
                "like": "1",
                "multimodalx_preference_probability": "0.10",
                "multimodalx2_preference_probability": "0.80",
            }
        ]
        components = [
            {
                "timestamp": "1",
                "screenshot": "a.jpg",
                "ridge": "10",
                "multimodal": "20",
                "knn_k9": "30",
                "knn_k11": "90",
                "regressor_path": "ridge_round3.joblib",
                "multimodal_regressor_path": "multi_round3.joblib",
            }
        ]

        rows = bumble_preference.veto_x3_formula_rows(manifest, components, formula=formula)

        self.assertEqual(rows[0]["score"], "46.000000")
        self.assertEqual(rows[0]["knn"], "30.000000")
        self.assertEqual(rows[0]["old_p_like"], "0.800000")
        self.assertIn("old_p_like", bumble_preference.veto_x3_feature_fields(formula))


def touch(path: Path) -> None:
    path.write_bytes(b"x")


def veto_manifest_row(
    selected_path: str,
    *,
    round3: str,
    multimodalx: str,
    multimodalx2: str,
    experimental1: str,
) -> dict[str, str]:
    return {
        "timestamp": selected_path,
        "screenshot": selected_path,
        "selected_path": selected_path,
        "disagreement_pattern": "",
        "round3_action": round3,
        "multimodalx_action": multimodalx,
        "multimodalx2_action": multimodalx2,
        "experimental1_action": experimental1,
    }


if __name__ == "__main__":
    unittest.main()
