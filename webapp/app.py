"""
71803 이상감지 모델을 mmWave 실시간 데이터에 붙인 웹 대시보드.

주의: 이상감지 모델(models/anomaly_classifier.joblib)은 AI Hub 71803의 10분
간격 데이터로 학습됐다. mmWave는 훨씬 빠르게 값을 낸다(초 단위). 그래서 롤링
피처(heart_rate_roll_mean_3 등)를 "최근 N개 샘플"이 아니라 "최근 30분/60분"
같은 실제 시간창으로 다시 계산한다 — 원래 피처가 30분/60분 평균을 의도한 거라,
이렇게 해야 학습 때와 같은 의미의 값이 된다.

수면단계 모델(models/sleep_stage_classifier.joblib)은 검증 정확도 34.5%로
참고용일 뿐이다. HRV 피처(hr_sdnn_5, hr_rmssd_5)는 원래 beat-to-beat RR
interval이 필요한데 mmWave는 평균 bpm만 주므로, 여기선 5분 윈도우 내 bpm
표준편차/제곱평균제곱근차분으로 근사한다 — 근사치라는 점을 UI에도 명시한다.

사용법:
  python3 webapp/app.py [--serial-port /dev/ttyACM0]
  브라우저에서 http://localhost:5050 접속
"""
import argparse
import csv
import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import serial
from flask import Flask, jsonify, render_template

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "webapp_log.csv"
)
LINE_RE = re.compile(r"(\w+):\s*(-?\d+\.?\d*)")

ANOMALY_FEATURES = [
    "heart_rate", "breath_rate",
    "heart_rate_diff", "heart_rate_roll_mean_3", "heart_rate_roll_std_3",
    "heart_rate_roll_mean_6", "heart_rate_roll_std_6",
    "breath_rate_diff", "breath_rate_roll_mean_3", "breath_rate_roll_std_3",
    "breath_rate_roll_mean_6", "breath_rate_roll_std_6",
]
SLEEP_FEATURES = [
    "heart_rate", "respiratory_rate",
    "hr_mean", "hr_sdnn_5", "hr_rmssd_5", "hr_slope_3",
    "rr_mean", "rr_sd_5", "rr_slope_3",
    "hr_rr_ratio", "hr_rr_product",
]

WINDOW_30MIN = 30 * 60
WINDOW_60MIN = 60 * 60
WINDOW_5MIN = 5 * 60
WINDOW_3MIN = 3 * 60
HISTORY_MAX = 720

app = Flask(__name__)
anomaly_model = joblib.load(os.path.join(MODELS_DIR, "anomaly_classifier.joblib"))
sleep_model = joblib.load(os.path.join(MODELS_DIR, "sleep_stage_classifier.joblib"))

state_lock = threading.Lock()
history = deque(maxlen=HISTORY_MAX)  # (epoch_sec, heart_rate, breath_rate, distance_cm, presence)
latest = {
    "heart_rate": None, "breath_rate": None, "distance_cm": None,
    "presence": 0, "anomaly_prob": None,
    "sleep_stage": None, "sleep_stage_prob": None,
    "connected": False, "updated_at": None,
}
daily = {"date": None, "hr_sum": 0.0, "br_sum": 0.0, "n": 0, "anomaly_count": 0}
prev_hr, prev_br = None, None
prev_anomaly_flag = False


def _init_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "heart_rate", "breath_rate", "distance_cm",
                 "presence", "anomaly_prob", "sleep_stage"]
            )


def _append_log(ts, hr, br, distance, presence, prob, stage):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            hr, br, distance, presence, prob, stage,
        ])


def _update_daily(ts, hr, br, is_anomaly):
    today = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    if daily["date"] != today:
        daily.update({"date": today, "hr_sum": 0.0, "br_sum": 0.0, "n": 0, "anomaly_count": 0})
    daily["hr_sum"] += hr
    daily["br_sum"] += br
    daily["n"] += 1
    if is_anomaly:
        daily["anomaly_count"] += 1


def windowed_vals(records, key, now_ts, window_sec):
    return [r[key] for r in records if now_ts - r["ts"] <= window_sec]


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0


def slope(vals):
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals))
    return float(np.polyfit(x, vals, 1)[0])


def rmssd(vals):
    if len(vals) < 2:
        return 0.0
    diffs = np.diff(vals)
    return float(np.sqrt(np.mean(diffs ** 2)))


def build_anomaly_features(records, now_ts, hr, br, hr_diff, br_diff):
    hr30 = windowed_vals(records, "heart_rate", now_ts, WINDOW_30MIN)
    hr60 = windowed_vals(records, "heart_rate", now_ts, WINDOW_60MIN)
    br30 = windowed_vals(records, "breath_rate", now_ts, WINDOW_30MIN)
    br60 = windowed_vals(records, "breath_rate", now_ts, WINDOW_60MIN)
    hr_mean3, hr_std3 = mean_std(hr30)
    hr_mean6, hr_std6 = mean_std(hr60)
    br_mean3, br_std3 = mean_std(br30)
    br_mean6, br_std6 = mean_std(br60)
    return pd.DataFrame([[
        hr, br, hr_diff, hr_mean3, hr_std3, hr_mean6, hr_std6,
        br_diff, br_mean3, br_std3, br_mean6, br_std6,
    ]], columns=ANOMALY_FEATURES)


def build_sleep_features(records, now_ts, hr, br):
    hr5 = windowed_vals(records, "heart_rate", now_ts, WINDOW_5MIN)
    hr3 = windowed_vals(records, "heart_rate", now_ts, WINDOW_3MIN)
    br5 = windowed_vals(records, "breath_rate", now_ts, WINDOW_5MIN)
    br3 = windowed_vals(records, "breath_rate", now_ts, WINDOW_3MIN)

    hr_mean, hr_sdnn5 = mean_std(hr5)
    rr_mean, rr_sd5 = mean_std(br5)
    hr_rmssd5 = rmssd(hr5)
    hr_slope3 = slope(hr3)
    rr_slope3 = slope(br3)
    hr_rr_ratio = hr / br if br else 0.0
    hr_rr_product = hr * br

    return pd.DataFrame([[
        hr, br, hr_mean, hr_sdnn5, hr_rmssd5, hr_slope3,
        rr_mean, rr_sd5, rr_slope3, hr_rr_ratio, hr_rr_product,
    ]], columns=SLEEP_FEATURES)


def serial_worker(port, baud):
    global prev_hr, prev_br, prev_anomaly_flag
    buf = {}
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=2)
            with state_lock:
                latest["connected"] = True
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                for key, value in LINE_RE.findall(line):
                    if key in ("heart_rate", "breath_rate"):
                        buf[key] = float(value)
                    elif key == "distance":
                        buf["distance_cm"] = float(value)

                if "heart_rate" in buf and "breath_rate" in buf:
                    now_ts = time.time()
                    hr, br = buf["heart_rate"], buf["breath_rate"]
                    distance = buf.get("distance_cm")
                    presence = 1 if distance is not None and distance > 0 else 0

                    with state_lock:
                        records = [
                            {"ts": t, "heart_rate": h, "breath_rate": b}
                            for t, h, b, _, _ in history
                        ]
                        hr_diff = 0.0 if prev_hr is None else hr - prev_hr
                        br_diff = 0.0 if prev_br is None else br - prev_br
                        prev_hr, prev_br = hr, br

                        x_anomaly = build_anomaly_features(records, now_ts, hr, br, hr_diff, br_diff)
                        prob = float(anomaly_model.predict_proba(x_anomaly)[0, 1])

                        x_sleep = build_sleep_features(records, now_ts, hr, br)
                        stage_pred = sleep_model.predict(x_sleep)[0]
                        stage_proba = sleep_model.predict_proba(x_sleep)[0]
                        stage_prob = float(max(stage_proba))

                        history.append((now_ts, hr, br, distance, presence))

                        is_anomaly = prob >= 0.5
                        newly_anomalous = is_anomaly and not prev_anomaly_flag
                        prev_anomaly_flag = is_anomaly

                        _update_daily(now_ts, hr, br, is_anomaly)
                        _append_log(now_ts, hr, br, distance, presence, round(prob, 3), stage_pred)

                        latest.update({
                            "heart_rate": hr, "breath_rate": br,
                            "distance_cm": distance, "presence": presence,
                            "anomaly_prob": round(prob, 3),
                            "newly_anomalous": newly_anomalous,
                            "sleep_stage": stage_pred,
                            "sleep_stage_prob": round(stage_prob, 3),
                            "connected": True,
                            "updated_at": datetime.now().strftime("%H:%M:%S"),
                            "daily_avg_hr": round(daily["hr_sum"] / daily["n"], 1) if daily["n"] else None,
                            "daily_avg_br": round(daily["br_sum"] / daily["n"], 1) if daily["n"] else None,
                            "daily_anomaly_count": daily["anomaly_count"],
                        })
                    buf = {}
        except (serial.SerialException, OSError):
            with state_lock:
                latest["connected"] = False
            time.sleep(2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with state_lock:
        hist = [
            {"t": t, "heart_rate": hr, "breath_rate": br}
            for t, hr, br, _, _ in list(history)[-180:]
        ]
        resp = {**latest, "history": hist}
        latest["newly_anomalous"] = False  # 1회성 신호이므로 읽고 나면 소비
        return jsonify(resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    _init_log()

    t = threading.Thread(
        target=serial_worker, args=(args.serial_port, args.baud), daemon=True
    )
    t.start()

    app.run(host="0.0.0.0", port=args.port, debug=False)
