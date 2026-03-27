import pytest

from deterministic_ai_agent.ner.extractor import NERExtractor


@pytest.fixture
def ner() -> NERExtractor:
    return NERExtractor()


# ---------------------------------------------------------------------------
# Device ID extraction
# ---------------------------------------------------------------------------


def test_extracts_conveyor_device_id(ner: NERExtractor):
    params = ner.extract("Conveyor_A has stopped unexpectedly.")
    assert params["device_id"] == "Conveyor_A"
    assert params["item_name"] == "Conveyor_A"


def test_extracts_motor_device_id(ner: NERExtractor):
    params = ner.extract("Motor_B shows signs of overheating.")
    assert params["device_id"] == "Motor_B"


def test_extracts_sensor_device_id(ner: NERExtractor):
    params = ner.extract("Sensor_C calibration drift detected.")
    assert params["device_id"] == "Sensor_C"


def test_extracts_pump_device_id(ner: NERExtractor):
    params = ner.extract("Pump_01 pressure drop detected.")
    assert params["device_id"] == "Pump_01"


# ---------------------------------------------------------------------------
# Sensor label fallback
# ---------------------------------------------------------------------------


def test_extracts_sensor_label_when_no_device_id(ner: NERExtractor):
    params = ner.extract("Sensor XYZ reading is 85.5. Threshold exceeded.")
    assert params.get("sensor") == "XYZ"
    assert "device_id" not in params


def test_sensor_label_is_case_insensitive(ner: NERExtractor):
    params = ner.extract("sensor abc threshold exceeded.")
    assert params.get("sensor") == "ABC"


# ---------------------------------------------------------------------------
# Line number extraction
# ---------------------------------------------------------------------------


def test_extracts_line_number(ner: NERExtractor):
    params = ner.extract("Critical failure on production line 3.")
    assert params["line_id"] == 3


def test_extracts_japanese_line_number(ner: NERExtractor):
    params = ner.extract("ライン2 の稼働状況を確認してください。")
    assert params["line_id"] == 2


# ---------------------------------------------------------------------------
# Numeric value extraction
# ---------------------------------------------------------------------------


def test_extracts_float_value(ner: NERExtractor):
    params = ner.extract("Sensor XYZ reading is 85.5.")
    assert params["value"] == pytest.approx(85.5)


def test_extracts_integer_value(ner: NERExtractor):
    params = ner.extract("Temperature reached 120°C.")
    assert params["value"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Fault keyword extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_fault",
    [
        ("High motor temperature overheat detected.", "overheat"),
        ("Pump_01 pressure drop alert.", "pressure_drop"),
        ("Conveyor_A vibration threshold exceeded.", "vibration"),
        ("Motor_B overcurrent protection triggered.", "overcurrent"),
        ("Sensor_C calibration drift found.", "calibration_drift"),
        ("Manual stop initiated by operator.", "manual_stop"),
    ],
)
def test_extracts_fault_keyword(ner: NERExtractor, text: str, expected_fault: str):
    params = ner.extract(text)
    assert params.get("fault") == expected_fault


# ---------------------------------------------------------------------------
# Unknown / empty input (safety: must not raise, must return empty dict)
# ---------------------------------------------------------------------------


def test_unknown_input_returns_empty_dict(ner: NERExtractor):
    params = ner.extract("This sentence contains no recognisable entities at all.")
    assert params == {}


def test_empty_string_returns_empty_dict(ner: NERExtractor):
    params = ner.extract("")
    assert params == {}


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------


def test_combined_device_line_fault(ner: NERExtractor):
    params = ner.extract("Motor_B overheat detected on production line 3. Temperature 95.2°C.")
    assert params["device_id"] == "Motor_B"
    assert params["line_id"] == 3
    assert params["fault"] == "overheat"
    assert params["value"] == pytest.approx(95.2)
