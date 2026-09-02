"""
XIAO ESP32C6 + MR60BHA2(mmwaveBreath 스케치)의 시리얼 출력을 읽어서
AI Hub 71803 데이터와 맞출 수 있는 공통 스키마로 CSV에 기록한다.

공통 스키마: timestamp, heart_rate, breath_rate, distance_cm, presence

사용법:
  python3 scripts/mmwave_logger.py [--port /dev/ttyACM0] [--out ../data/processed/mmwave_log.csv]

주의: 시리얼 포트 권한이 없으면 dialout 그룹에 속해있는지 확인할 것
  (sudo usermod -aG dialout $USER 후 재로그인)
"""
import argparse
import csv
import os
import re
import time

import serial

LINE_RE = re.compile(r"(\w+):\s*(-?\d+\.?\d*)")


def parse_line(line, state):
    """한 줄에서 key: value 쌍을 찾아 state에 누적한다."""
    for key, value in LINE_RE.findall(line):
        if key in ("heart_rate", "breath_rate"):
            state[key] = float(value)
        elif key == "distance":
            state["distance_cm"] = float(value)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    default_out = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "mmwave_log.csv"
    )
    parser.add_argument("--out", default=default_out)
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=1.0,
        help="이 간격(초)마다 누적된 값을 한 행으로 CSV에 기록",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    file_exists = os.path.exists(args.out)

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"{args.port} @ {args.baud}bps 연결됨. Ctrl+C로 종료.")

    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "heart_rate", "breath_rate", "distance_cm", "presence"]
            )

        state = {}
        last_flush = time.time()

        try:
            while True:
                raw = ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line:
                        parse_line(line, state)

                now = time.time()
                if now - last_flush >= args.flush_interval:
                    if state:
                        presence = 1 if "distance_cm" in state else 0
                        row = [
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            state.get("heart_rate", ""),
                            state.get("breath_rate", ""),
                            state.get("distance_cm", ""),
                            presence,
                        ]
                        writer.writerow(row)
                        f.flush()
                        print(row)
                        state = {}
                    last_flush = now
        except KeyboardInterrupt:
            print("\n종료합니다.")


if __name__ == "__main__":
    main()
