import typing


def run_diagnostics(device_id: str) -> dict[str, typing.Any]:
    """
    Simulated tool to run diagnostics on a specific machine.
    """
    # Dummy logic to simulate a real diagnostic tool.
    diagnostics_data = {
        "Conveyor_A": {
            "health": "Poor",
            "last_check": "2026-03-27",
            "details": "High vibration detected.",
        },
        "Motor_B": {
            "health": "Excellent",
            "last_check": "2026-03-26",
            "details": "Running within normal range.",
        },
        "Sensor_C": {
            "health": "Fair",
            "last_check": "2026-03-25",
            "details": "Slight calibration drift.",
        },
    }

    result = diagnostics_data.get(
        device_id, {"health": "Unknown", "last_check": "N/A", "details": "No data available."}
    )
    return {"tool": "run_diagnostics", "device": device_id, "result": result}
