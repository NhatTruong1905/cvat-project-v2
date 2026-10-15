import unittest

from cvat.apps.model_assisted_qc.benchmark.evaluate import to_coco_ground_truth
from cvat.apps.model_assisted_qc.benchmark.run_benchmark import normalize_label


class BenchmarkHelpersTest(unittest.TestCase):
    def test_normalize_label_maps_bdd_aliases(self):
        classes = ["pedestrian", "traffic sign"]
        self.assertEqual(normalize_label("person", classes), "pedestrian")
        self.assertEqual(normalize_label("stop sign", classes), "traffic sign")
        self.assertIsNone(normalize_label("cat", classes))

    def test_ground_truth_boxes_become_coco_xywh(self):
        frames = [{
            "filename": "one.jpg", "width": 100, "height": 80,
            "boxes": [{"class": "car", "bbox": [10, 20, 40, 60]}],
        }]
        result = to_coco_ground_truth(frames, ["car"])
        self.assertEqual(result["annotations"][0]["bbox"], [10, 20, 30, 40])
        self.assertEqual(result["annotations"][0]["area"], 1200)


if __name__ == "__main__":
    unittest.main()
