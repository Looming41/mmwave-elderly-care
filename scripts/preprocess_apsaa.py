"""
APSAA(Audio-Polygraphy Dataset for Sleep Apnea Analysis, Zenodo 14096541)를
1분 단위 피처 테이블로 변환한다.

Apnea-ECG와 다른 점: 여기는 심박수가 이미 bpm으로 계산돼 있고(Pulse_EG, 3Hz),
흉부 호흡 신호(Thorax_EG, 50Hz)가 따로 있어서 호흡수를 ECG 유도 없이 직접
추정할 수 있다 — mmWave(심박수+호흡수 둘 다 직접 출력)에 훨씬 가까운 조건이다.

라벨: Obstructive/Central/Mixed Apnea + Hypopnea 이벤트가 겹치는 분을 1로 표시
(PhysioNet apnea-ecg와 동일한 관례: 무호흡+저호흡을 묶어서 "무호흡 분"으로 정의).

원본 데이터는 재배포 금지라 프로젝트에 복사하지 않고 경로만 참조한다.

사용법:
  python3 scripts/preprocess_apsaa.py [--data-dir ~/다운로드/APSAA_extracted]
  -> data/processed/apsaa_minutes.csv
"""
import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd

PULSE_HZ = 3.0
THORAX_HZ = 50.0
APNEA_EVENTS = {"Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"}

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "apsaa_minutes.csv"
)


def to_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_annotations(path):
    events = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Event_Name"] in APNEA_EVENTS:
                start = to_sec(row["Start_Time"])
                events.append((start, start + float(row["Duration"])))
    return events


def load_signal(path):
    with open(path) as f:
        f.readline()  # header "Signal"
        return np.array([float(x) for x in f if x.strip()])


def breath_rate_from_thorax(seg):
    """1분(50Hz) 흉부 신호에서 0.1~0.5Hz 대역 피크 주파수를 호흡수(회/분)로."""
    if len(seg) < 200:
        return np.nan
    seg = seg - seg.mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), d=1 / THORAX_HZ)
    band = (freqs >= 0.1) & (freqs <= 0.5)
    if not band.any():
        return np.nan
    return float(freqs[band][np.argmax(spec[band])] * 60)


def process_subject(subject_dir):
    name = os.path.basename(subject_dir.rstrip("/"))
    pulse_path = os.path.join(subject_dir, f"{name}_Pulse_EG.csv")
    thorax_path = os.path.join(subject_dir, f"{name}_Thorax_EG.csv")
    ann_path = os.path.join(subject_dir, f"{name}_Annotations.csv")
    if not (os.path.exists(pulse_path) and os.path.exists(thorax_path) and os.path.exists(ann_path)):
        return []

    pulse = load_signal(pulse_path)
    thorax = load_signal(thorax_path)
    events = load_annotations(ann_path)

    n_minutes = int(len(pulse) / PULSE_HZ // 60)
    labels = np.zeros(n_minutes, dtype=int)
    for start, end in events:
        m0, m1 = int(start // 60), int(end // 60)
        for m in range(max(m0, 0), min(m1 + 1, n_minutes)):
            labels[m] = 1

    rows = []
    for m in range(n_minutes):
        p_lo, p_hi = int(m * 60 * PULSE_HZ), int((m + 1) * 60 * PULSE_HZ)
        t_lo, t_hi = int(m * 60 * THORAX_HZ), int((m + 1) * 60 * THORAX_HZ)
        p_seg = pulse[p_lo:p_hi]
        t_seg = thorax[t_lo:t_hi]
        p_seg = p_seg[(p_seg > 20) & (p_seg < 220)]  # 생리적으로 말이 안 되는 값 제거
        if len(p_seg) < 10:
            continue
        rows.append({
            "subject": name,
            "minute": m,
            "hr_mean": float(p_seg.mean()),
            "hr_min": float(p_seg.min()),
            "hr_max": float(p_seg.max()),
            "hr_std": float(p_seg.std()),
            "breath_rate_est": breath_rate_from_thorax(t_seg),
            "label": int(labels[m]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=os.path.expanduser("~/다운로드/APSAA_extracted")
    )
    args = parser.parse_args()

    subject_dirs = sorted(d for d in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(d))
    print(f"대상자 {len(subject_dirs)}명 처리 중...")

    all_rows = []
    for d in subject_dirs:
        all_rows.extend(process_subject(d))

    df = pd.DataFrame(all_rows).dropna(subset=["hr_mean", "breath_rate_est"])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"총 {len(df)}분, 대상자 {df['subject'].nunique()}명 -> {OUT_PATH}")
    print(f"무호흡+저호흡 비율: {df['label'].mean():.2%}")
    print(df[["hr_mean", "breath_rate_est"]].describe().round(2))


if __name__ == "__main__":
    main()
