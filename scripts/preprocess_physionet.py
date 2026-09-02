"""
PhysioNet Challenge 2019(패혈증 조기예측) .psv 파일들을 정리한다.

각 파일 = 환자 1명, 시간(hour) 단위 행. 우리가 쓸 건 HR, Resp, SepsisLabel뿐
(다른 컬럼은 mmWave가 못 주는 값이라 제외).

결측치는 환자 내에서 forward-fill(직전 값 유지) 후, 그래도 없는 초반 행은 버린다
(실제 배포 때도 mmWave가 첫 측정 전엔 값이 없는 것과 같은 상황이라 자연스러운 처리).

사용법:
  python3 scripts/preprocess_physionet.py
  -> data/processed/physionet2019_labeled.csv
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "physionet2019", "training_setA"
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "physionet2019_labeled.csv"
)


def load_patient(path):
    df = pd.read_csv(path, sep="|")
    df = df[["HR", "Resp", "ICULOS", "SepsisLabel"]].copy()
    df["HR"] = df["HR"].ffill()
    df["Resp"] = df["Resp"].ffill()
    df["patient_id"] = os.path.basename(path).replace(".psv", "")
    return df


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.psv")))
    print(f"환자 파일 {len(files)}개 처리 중...")

    frames = []
    for fp in files:
        try:
            frames.append(load_patient(fp))
        except Exception as e:
            print(f"[스킵] {fp}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["HR", "Resp"])
    df = df.rename(columns={
        "HR": "heart_rate", "Resp": "breath_rate", "ICULOS": "hours_since_admit",
    })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"총 {len(df)}행, 환자 {df['patient_id'].nunique()}명 -> {OUT_PATH}")
    print(f"SepsisLabel 양성 비율: {df['SepsisLabel'].mean():.2%}")


if __name__ == "__main__":
    main()
