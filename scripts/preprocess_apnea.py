"""
PhysioNet Apnea-ECG Database(수면 무호흡)를 1분 단위 피처 테이블로 변환한다.

원본: ECG 100Hz, 박동별 R피크(.qrs), 1분 단위 무호흡 라벨(.apn: N=정상, A=무호흡).
mmWave가 실시간으로 줄 수 있는 값에 맞춰서 R피크(박동 간격)에서 다음을 계산:
  - hr_mean/hr_min/hr_max : 1분 안 심박수 (bpm)
  - rmssd                 : 실제 박동간 간격 기반 심박변이도 (ms)
  - rr_std                : 박동간 간격 표준편차 (ms)
  - breath_rate_est       : RR 간격 변동의 주파수(0.1~0.5Hz) 피크 → 호흡수 추정 (회/분)
    ※ ECG 기반 호흡 추정치이고, mmWave의 실제 호흡수와는 측정 원리가 다름

원본 데이터 폴더는 재배포 대상이 아니므로 프로젝트에 복사하지 않고 경로만 참조한다.

사용법:
  python3 scripts/preprocess_apnea.py [--data-dir ~/다운로드/apnea-ecg-database-1.0.0]
  -> data/processed/apnea_minutes.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import wfdb

FS = 100
SAMPLES_PER_MIN = FS * 60
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "apnea_minutes.csv"
)


def rr_breath_rate(rr_s, t_s):
    """1분 구간의 RR 시계열을 4Hz로 보간해 0.1~0.5Hz 피크 주파수를 호흡수(회/분)로 변환."""
    if len(rr_s) < 8:
        return np.nan
    tt = np.arange(t_s[0], t_s[-1], 0.25)
    if len(tt) < 16:
        return np.nan
    rr_interp = np.interp(tt, t_s, rr_s)
    rr_interp = rr_interp - rr_interp.mean()
    spec = np.abs(np.fft.rfft(rr_interp * np.hanning(len(rr_interp))))
    freqs = np.fft.rfftfreq(len(rr_interp), d=0.25)
    band = (freqs >= 0.1) & (freqs <= 0.5)
    if not band.any():
        return np.nan
    return float(freqs[band][np.argmax(spec[band])] * 60)


def process_record(data_dir, rec_name):
    rec_path = os.path.join(data_dir, rec_name)
    qrs = wfdb.rdann(rec_path, "qrs").sample
    apn = wfdb.rdann(rec_path, "apn")
    n_minutes = int(np.ceil(
        wfdb.rdheader(rec_path).sig_len / SAMPLES_PER_MIN
    ))

    labels = np.zeros(n_minutes, dtype=int)
    for s, sym in zip(apn.sample, apn.symbol):
        m = int(s // SAMPLES_PER_MIN)
        if m < n_minutes and sym == "A":
            labels[m] = 1

    rr_ms = np.diff(qrs) / FS * 1000.0
    beat_t_s = qrs[1:] / FS
    valid = (rr_ms > 300) & (rr_ms < 2000)
    rr_ms, beat_t_s = rr_ms[valid], beat_t_s[valid]

    rows = []
    for m in range(n_minutes):
        lo, hi = m * 60.0, (m + 1) * 60.0
        sel = (beat_t_s >= lo) & (beat_t_s < hi)
        rr_m, t_m = rr_ms[sel], beat_t_s[sel]
        if len(rr_m) < 10:
            continue
        hr = 60000.0 / rr_m
        rmssd = float(np.sqrt(np.mean(np.diff(rr_m) ** 2))) if len(rr_m) > 1 else np.nan
        rows.append({
            "record": rec_name,
            "minute": m,
            "hr_mean": float(hr.mean()),
            "hr_min": float(hr.min()),
            "hr_max": float(hr.max()),
            "rr_std": float(rr_m.std()),
            "rmssd": rmssd,
            "breath_rate_est": rr_breath_rate(rr_m, t_m),
            "label": int(labels[m]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/다운로드/apnea-ecg-database-1.0.0"),
    )
    args = parser.parse_args()

    # 라벨이 있는 본 기록만: a01~a20, b01~b05, c01~c10 (r/er 변형, x 테스트셋 제외)
    labeled = sorted(
        os.path.basename(p).replace(".apn", "")
        for p in glob.glob(os.path.join(args.data_dir, "*.apn"))
        if re.fullmatch(r"[abc]\d{2}", os.path.basename(p).replace(".apn", ""))
    )
    print(f"라벨 있는 기록 {len(labeled)}개 처리 중...")

    all_rows = []
    for rec in labeled:
        all_rows.extend(process_record(args.data_dir, rec))

    df = pd.DataFrame(all_rows).dropna(subset=["hr_mean", "rmssd"])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"총 {len(df)}분, 기록 {df['record'].nunique()}개 -> {OUT_PATH}")
    print(f"무호흡(A) 비율: {df['label'].mean():.2%}")
    print(df[["hr_mean", "rmssd", "breath_rate_est"]].describe().round(2))


if __name__ == "__main__":
    main()
