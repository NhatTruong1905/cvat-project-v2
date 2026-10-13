from unittest import TestCase

from ..qc_core import frame_score, rank_frames, score_frame


class RiskScoringTest(TestCase):
    def test_detects_drop_and_flip_as_independent_risks(self):
        humans = [{"id": "10", "label": "car", "bbox": [0, 0, 100, 100]}]
        predictions = [
            {"external_id": "p1", "label": "truck", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
            {"external_id": "p2", "label": "person", "confidence": 0.8, "bbox": [200, 200, 240, 280]},
        ]

        risks = score_frame(humans, predictions)

        self.assertEqual([risk["type"] for risk in risks], ["flip", "drop"])
        self.assertAlmostEqual(risks[0]["score"], 0.95)
        self.assertAlmostEqual(risks[1]["score"], 0.8)

    def test_pd_scores_each_human_box_without_hungarian_assignment(self):
        humans = [
            {"id": "1", "label": "car", "bbox": [0, 0, 100, 100]},
            {"id": "2", "label": "car", "bbox": [5, 5, 95, 95]},
        ]
        predictions = [{"external_id": "p1", "label": "car", "confidence": 0.9, "bbox": [0, 0, 100, 100]}]

        risks = score_frame(humans, predictions)

        self.assertEqual([risk["type"] for risk in risks], ["ok", "ok"])
        self.assertTrue(all(risk["score_pd"] < 0.5 for risk in risks))

    def test_spawn_uses_pd(self):
        risks = score_frame([{"id": "1", "label": "car", "bbox": [0, 0, 100, 100]}], [])
        self.assertEqual(risks[0]["type"], "spawn")
        self.assertEqual(risks[0]["score"], 0.5)

    def test_top_fifteen_percent_is_frame_level(self):
        scores = {frame: 0.0 for frame in range(20)}
        scores.update({1: frame_score([0.94, 0.81, 0.5]), 2: frame_score([0.5] * 5), 12: 1.795})
        order, selected = rank_frames(scores, 0.15)
        self.assertEqual(order[:3], [2, 1, 12])
        self.assertEqual(selected, {1, 2, 12})
