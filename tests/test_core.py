import unittest

from nyc311_shadow.core import ShadowService, make_features


GOOD = {
    "complaint_type": "HEAT/HOT WATER",
    "borough": "BRONX",
    "created_date": "2026-09-26T10:00:00",
}


class FakeEncoder:
    categories_ = [["HEAT/HOT WATER", "Noise - Residential"]]


class FakeModel:
    named_steps = {"prepare": FakeEncoder()}

    def predict_proba(self, frame):
        assert list(frame.columns) == ["complaint_type", "borough", "month", "hour", "weekday"]
        return [[0.75, 0.25]]


class ContractTests(unittest.TestCase):
    def test_features_are_creation_time_only(self):
        row = make_features(GOOD).iloc[0]
        self.assertEqual((row["month"], row["hour"], row["weekday"]), (9, 10, 5))

    def test_rejects_leaked_and_missing_fields(self):
        for bad in ({**GOOD, "closed_date": "2026-09-27T10:00:00"},
                    {"borough": "BRONX", "created_date": GOOD["created_date"]}):
            with self.assertRaises(ValueError):
                make_features(bad)

    def test_rejects_invalid_local_times(self):
        for timestamp in ("not a date", "2026-03-08T02:30:00", "2026-11-01T01:30:00",
                          "2026-09-26T10:00:00+00:00"):
            with self.assertRaises(ValueError):
                make_features({**GOOD, "created_date": timestamp})

    def test_shadow_response_carries_lineage_and_warning(self):
        service = ShadowService(FakeModel(), {"threshold": 0.2,
                                             "final_evaluation_id": "example"})
        result = service.predict(GOOD)
        self.assertEqual(result["predicted_label"], 1)
        self.assertEqual(result["model_id"], "example")
        self.assertEqual(result["operational_status"], "HOLD_FOR_OPERATIONAL_USE")
        self.assertIsNotNone(result["review_warning"])


if __name__ == "__main__":
    unittest.main()

