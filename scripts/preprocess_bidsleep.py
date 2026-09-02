"""
BIDSleep(PhysioNet) 애플워치 심박수 + EEG(Dreem-2) 수면단계 라벨을 정리한다.

hr.csv(약 0.2Hz)를 30초 에폭 단위로 집계해서 expert_label(전문가 검수 라벨,
골드스탠다드)과 매칭한다. motion.csv(가속도계)는 mmWave가 못 주는 값이라 사용 안 함.

라벨 인코딩 (README 기준): Wake=0, N1=1, N2=2, N3=3, REM=4, Unknown=5(제외)

사용법:
  python3 scripts/preprocess_bidsleep.py
  -> data/processed/bidsleep_labeled.csv
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.io

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "bidsleep")
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "bidsleep_labeled.csv"
)

STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def process_night(night_dir):
    hr_path = os.path.join(night_dir, "hr.csv")
    label_path = os.path.join(night_dir, "labels.mat")
    if not (os.path.exists(hr_path) and os.path.exists(label_path)):
        return None

    hr = pd.read_csv(hr_path, header=None, names=["ts", "hr"])
    mat = scipy.io.loadmat(label_path)
    labels = mat["expert_label"][0]
    rec_start = (
        pd.Timestamp(str(mat["recStart"][0]))
        .tz_localize("America/New_York")
        .timestamp()
    )

    hr["epoch"] = ((hr["ts"] - rec_start) // 30).astype(int)
    hr = hr[(hr["epoch"] >= 0) & (hr["epoch"] < len(labels))]

    agg = hr.groupby("epoch")["hr"].agg(["mean", "std", "min", "max", "count"])
    agg.columns = ["hr_mean", "hr_std", "hr_min", "hr_max", "n_samples"]
    agg = agg.reset_index()
    agg["stage_code"] = labels[agg["epoch"].values]

    parts = night_dir.rstrip("/").split(os.sep)
    agg["subject"] = parts[-2]
    agg["night"] = parts[-1]
    return agg


def main():
    night_dirs = sorted(glob.glob(os.path.join(RAW_DIR, "Bidslab*", "*")))
    night_dirs = [d for d in night_dirs if os.path.isdir(d)]
    print(f"{len(night_dirs)}개 야간 폴더 처리 중...")

    frames = []
    for nd in night_dirs:
        r = process_night(nd)
        if r is not None:
            frames.append(r)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["stage_code"].isin(STAGE_MAP.keys())].copy()
    df["stage"] = df["stage_code"].map(STAGE_MAP)
    df["hr_std"] = df["hr_std"].fillna(0)
    df = df.sort_values(["subject", "night", "epoch"])
    df["hr_diff"] = df.groupby(["subject", "night"])["hr_mean"].diff().fillna(0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"총 {len(df)}행, 대상자 {df['subject'].nunique()}명, "
          f"{df.groupby('subject')['night'].nunique().sum()}박 -> {OUT_PATH}")
    print("\nstage 분포:")
    print(df["stage"].value_counts())


if __name__ == "__main__":
    main()
