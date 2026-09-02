"""
aihub_71803_labeled.csv에 롤링 통계 피처를 추가한다.

10분 간격 데이터이므로:
  - *_diff: 직전 측정값 대비 변화량 (급격한 변화 탐지)
  - *_roll_mean_3 / *_roll_std_3: 최근 3개(30분) 이동평균/표준편차
  - *_roll_mean_6 / *_roll_std_6: 최근 6개(60분) 이동평균/표준편차

subject_id별로 시간순 정렬 후 그룹 단위로 계산한다 (사람 간 데이터가 섞이면 안 됨).
mmWave 배포 시에도 최근 N개 샘플을 버퍼에 들고 있으면 동일하게 재현 가능한 피처들이다.

사용법:
  python3 scripts/feature_engineering.py
  -> data/processed/aihub_71803_features.csv 생성
"""
import os

import pandas as pd

IN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "aihub_71803_labeled.csv"
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "aihub_71803_features.csv"
)

BASE_COLS = ["heart_rate", "breath_rate"]
WINDOWS = [3, 6]


def add_rolling_features(group):
    group = group.sort_values("timestamp")
    for col in BASE_COLS:
        group[f"{col}_diff"] = group[col].diff()
        for w in WINDOWS:
            group[f"{col}_roll_mean_{w}"] = (
                group[col].rolling(w, min_periods=1).mean()
            )
            group[f"{col}_roll_std_{w}"] = (
                group[col].rolling(w, min_periods=2).std()
            )
    return group


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df = df.groupby("subject_id", group_keys=False)[df.columns].apply(add_rolling_features)

    # 롤링 std는 그룹 시작 부분에서 NaN이 나올 수 있음 -> 0으로 채움
    roll_std_cols = [c for c in df.columns if c.endswith("_std_3") or c.endswith("_std_6")]
    df[roll_std_cols] = df[roll_std_cols].fillna(0)
    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    df[diff_cols] = df[diff_cols].fillna(0)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"총 {len(df)}행, 컬럼 {len(df.columns)}개 -> {OUT_PATH}")
    print("\n추가된 피처:")
    new_cols = [c for c in df.columns if any(c.startswith(b) and c != b for b in BASE_COLS)]
    print(new_cols)


if __name__ == "__main__":
    main()
