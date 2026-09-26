import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from nyc311_shadow.api import app


class FakeService:
    release = {
        "final_evaluation_id": "example",
        "operational_status": "HOLD_FOR_OPERATIONAL_USE",
    }

    def predict(self, payload):
        if payload["created_date"] == "bad":
            raise ValueError("Invalid NYC created_date")
        return {"model_id": "example", "mode": "shadow", "probability_late": 0.25}


class ApiTests(unittest.TestCase):
    def test_health_prediction_and_rejected_leak(self):
        with patch("nyc311_shadow.api.load_release", return_value=FakeService()):
            with TestClient(app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["operational_status"], "HOLD_FOR_OPERATIONAL_USE")
                item = {"complaint_type": "HEAT/HOT WATER", "borough": "BRONX",
                        "created_date": "2026-09-26T10:00:00"}
                self.assertEqual(client.post("/predict", json=item).status_code, 200)
                self.assertEqual(client.post("/predict", json={**item, "closed_date": "late"}).status_code, 422)
                self.assertEqual(client.post("/predict", json={**item, "created_date": "bad"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()

