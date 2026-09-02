"""
Hugging Face abmallick/heart-breath-sleep-stage-dataset로 수면단계 분류 모델 학습.

mmWave가 밤새 심박수/호흡수를 기록한다면 얻을 수 있는 피처만 사용한다
(모두 heart_rate/respiratory_rate에서 파생된 값 — 다른 센서 불필요):
  heart_rate, respiratory_rate, hr_mean, hr_sdnn_5, hr_rmssd_5, hr_slope_3,
  rr_mean, rr_sd_5, rr_slope_3, hr_rr_ratio, hr_rr_product

night_id 기준으로 데이터셋이 이미 train/val/test로 나뉘어 있음 (person-disjoint).
sleep_stage 코드: 0=Wake, 1=N1, 2=N2, 3=N3, 5=REM (6, 9는 희귀/미채점이라 제외)

사용법:
  python3 scripts/train_sleep_stage_model.py
"""
import os

import joblib
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "sleep_stage_classifier.joblib"
)

FEATURES = [
    "heart_rate", "respiratory_rate",
    "hr_mean", "hr_sdnn_5", "hr_rmssd_5", "hr_slope_3",
    "rr_mean", "rr_sd_5", "rr_slope_3",
    "hr_rr_ratio", "hr_rr_product",
]

STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}


def main():
    ds = load_dataset("abmallick/heart-breath-sleep-stage-dataset")["train"]
    df = ds.to_pandas()
    df = df[df["sleep_stage"].isin(STAGE_MAP.keys())].copy()
    df["stage"] = df["sleep_stage"].map(STAGE_MAP)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    print(f"Train: {len(train_df)}행 / {train_df['night_id'].nunique()}밤")
    print(f"Val:   {len(val_df)}행 / {val_df['night_id'].nunique()}밤\n")
    print("Train stage 분포:")
    print(train_df["stage"].value_counts())

    X_train, y_train = train_df[FEATURES], train_df["stage"]
    X_val, y_val = val_df[FEATURES], val_df["stage"]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("\n=== 분류 리포트 (Validation) ===")
    print(classification_report(y_val, y_pred, digits=3))
    print("혼동행렬:")
    labels = sorted(y_val.unique())
    print(labels)
    print(confusion_matrix(y_val, y_pred, labels=labels))

    print("\n피처 중요도:")
    for name, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
