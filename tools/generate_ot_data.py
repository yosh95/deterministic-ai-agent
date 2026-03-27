import json
import random
from pathlib import Path


def generate_ot_data(num_samples=100):
    intents = [
        {
            "id": 0,
            "name": "DIAGNOSTICS",
            "templates": [
                "Critical: {device} reports {fault} (Code {code}). Request diagnostics.",
                "Alert: {device} {fault} detected on Line {line}. Check status.",
                "High temperature alarm on {device}. Run health check.",
                "{device} の {fault} アラート。診断を開始してください。",
                "システム異常：{device} の通信エラー (Code {code})。再起動と診断が必要。",
                "Warning: {device} vibration exceeds threshold ({value} mm/s).",
                "Emergency stop triggered on {line} due to {device} {fault}.",
            ],
        },
        {
            "id": 1,
            "name": "INVENTORY",
            "templates": [
                "Check stock for {device} spare parts.",
                "Do we have any {device} filters in the warehouse?",
                "Inventory status for {device} - procurement needs count.",
                "{device} の在庫はありますか？",
                "交換用の {device} ベルトを 2つ発注したい。在庫確認願います。",
                "Supply check: spare {device} units for {line}.",
                "Is there a replacement {device} available for immediate swap?",
            ],
        },
        {
            "id": 2,
            "name": "LOG_EVENT",
            "templates": [
                "Log: {device} manual stop by operator {operator}.",
                "Event: {device} started successfully on Line {line}.",
                "Routine maintenance completed for {device} (Task {task_id}).",
                "System nominal on Line {line}. No issues reported.",
                "{device} の定期点検完了。ログに記録します。",
                "Operator {operator}: {device} restarted after {reason}.",
                "Shift report: {line} throughput was {value} units/hr.",
            ],
        },
    ]

    devices = [
        "Motor_A",
        "Motor_B",
        "Conveyor_1",
        "Conveyor_2",
        "Pump_X",
        "Pump_Y",
        "Sensor_Z",
        "PLC_01",
        "Robotic_Arm_03",
    ]
    faults = [
        "overheat",
        "vibration",
        "low_pressure",
        "communication_error",
        "belt_slip",
        "calibration_drift",
    ]
    operators = ["Sato", "Tanaka", "Suzuki", "Ito", "Watanabe"]
    reasons = ["belt_tensioning", "filter_cleaning", "software_update", "manual_inspection"]

    data = []
    for i in range(num_samples):
        intent = random.choice(intents)
        template = random.choice(intent["templates"])

        device = random.choice(devices)
        fault = random.choice(faults)
        line = random.randint(1, 10)
        code = f"E{random.randint(100, 999)}"
        value = round(random.uniform(0.1, 100.0), 2)
        operator = random.choice(operators)
        task_id = f"T-{random.randint(1000, 9999)}"
        reason = random.choice(reasons)

        text = template.format(
            device=device,
            fault=fault,
            line=line,
            code=code,
            value=value,
            operator=operator,
            task_id=task_id,
            reason=reason,
        )

        params = {"device": device}
        if "fault" in template:
            params["fault"] = fault
        if "line" in template:
            params["line"] = line

        data.append({"id": i + 1, "input": text, "intent_id": intent["id"], "parameters": params})

    return data


def main():
    data = generate_ot_data(500)
    data_path = Path("data/ot_domain_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Generated {len(data)} OT domain samples in {data_path}")


if __name__ == "__main__":
    main()
