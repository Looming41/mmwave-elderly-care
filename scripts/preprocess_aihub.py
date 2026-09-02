"""
AI Hub 71803(독거노인 돌봄용 위험감지 데이터) 라벨 JSON을 학습용 테이블로 변환.

실제 확인된 스키마 (2026-08-26 다운로드 기준):
  TimeSeriesData[].SM_Sensor: HeartRate, BreathRate, SPO2, SkinTemperature,
                               SleepPhase, SleepScore, WalkingSteps,
                               StressIndex, ActivityIntensity, CaloricExpenditure
  TimeSeriesData[].Total_Labeling.Estimation: 기타 / 주의 / 수면 / 외출 / 위험 / 식사

mmWave(MR60BHA2) 센서가 실시간으로 낼 수 있는 값은 heart_rate, breath_rate,
presence(거리 유효 여부) 뿐이므로, 학습 피처는 두 그룹으로 나눠서 저장한다:
  - mmwave_compatible: heart_rate, breath_rate  (실제 배포 시 쓸 수 있는 피처)
  - aihub_only: spo2, skin_temp, sleep_phase, sleep_score, steps,
                stress_index, activity_intensity, calories (AI Hub에만 있는 피처,
                사전학습/보조정보로만 활용 가능)

사용법:
  python3 scripts/preprocess_aihub.py
  -> data/processed/aihub_71803_labeled.csv 생성
"""
import glob
import json
import os

import pandas as pd

RAW_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw",
    "68.독거노인_돌봄용_위험감지_데이터", "3.개방데이터", "1.데이터", "extracted",
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "aihub_71803_labeled.csv"
)


def load_label_files():
    files = glob.glob(os.path.join(RAW_ROOT, "TL_label", "*.json"))
    files += glob.glob(os.path.join(RAW_ROOT, "VL_label", "*.json"))
    return files


def json_to_rows(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    meta = d.get("MetaData", {})
    subject_id = meta.get("ID")
    age = meta.get("Age")
    gender = meta.get("Gender")

    rows = []
    for entry in d.get("TimeSeriesData", []):
        sm = entry.get("SM_Sensor", {})
        total = entry.get("Total_Labeling", {})
        if not sm:
            continue
        rows.append({
            "subject_id": subject_id,
            "age": age,
            "gender": gender,
            "timestamp": entry.get("TimeStamp"),
            "heart_rate": sm.get("HeartRate"),
            "breath_rate": sm.get("BreathRate"),
            "spo2": sm.get("SPO2"),
            "skin_temp": sm.get("SkinTemperature"),
            "sleep_phase": sm.get("SleepPhase"),
            "sleep_score": sm.get("SleepScore"),
            "steps": sm.get("WalkingSteps"),
            "stress_index": sm.get("StressIndex"),
            "activity_intensity": sm.get("ActivityIntensity"),
            "calories": sm.get("CaloricExpenditure"),
            "label": total.get("Estimation"),
        })
    return rows


def main():
    files = load_label_files()
    print(f"라벨 파일 {len(files)}개 처리 중...")

    all_rows = []
    for fp in files:
        all_rows.extend(json_to_rows(fp))

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["heart_rate", "breath_rate", "label"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"총 {len(df)}행 저장 -> {OUT_PATH}")
    print("\n라벨 분포:")
    print(df["label"].value_counts())
    print("\nheart_rate / breath_rate 기술통계:")
    print(df[["heart_rate", "breath_rate"]].describe())


if __name__ == "__main__":
    main()
