import unittest

from src.ner.extractor import NERExtractor


class TestNERHardened(unittest.TestCase):
    def setUp(self):
        self.ner = NERExtractor()

    def test_date_time_exclusion(self):
        # Input with date and time should NOT extract 2026 or 14:32 as value
        text = "2026-03-27 14:32:00 Motor_B temperature is 95.5°C on line 2."
        params = self.ner.extract(text)

        self.assertEqual(params.get("device_id"), "Motor_B")
        self.assertEqual(params.get("value"), 95.5)
        self.assertEqual(params.get("line_id"), 2)
        # Verify 2026 or 14 wasn't picked up as value
        self.assertNotEqual(params.get("value"), 2026.0)

    def test_device_from_config(self):
        # Test a device defined in config/devices.yaml
        text = "Checking status for Pump_01."
        params = self.ner.extract(text)
        self.assertEqual(params.get("device_id"), "Pump_01")

    def test_line_number_shadowing(self):
        # Line number should not be picked up as the primary value
        text = "Line 4 report: Sensor_C reading is 0.85 bar."
        params = self.ner.extract(text)
        self.assertEqual(params.get("line_id"), 4)
        self.assertEqual(params.get("value"), 0.85)

    def test_prefix_matching(self):
        # Test prefix + alphanumeric suffix (e.g. Motor_X123)
        text = "Alert on Motor_X123."
        params = self.ner.extract(text)
        self.assertEqual(params.get("device_id"), "Motor_X123")


if __name__ == "__main__":
    unittest.main()
