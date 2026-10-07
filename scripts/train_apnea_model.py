"""
Apnea-ECG(PhysioNet)로 "1분 단위 수면 무호흡 여부" 분류 모델 학습.

mmWave가 실시간으로 낼 수 있는 값(심박수, 호흡수, 심박변이도)만으로 무호흡을
맞힐 수 있는지 검증한다. 롤링 피처는 과거 값만 사용(실시간 배포와 동일한 조건).
기록(record) 단위로 train/test를 나눠서 같은 사람 데이터가 섞이지 않게 한다.

사용법:
  python3 scripts/train_apnea_model.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "apnea_minutes.csv"
)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "apnea_minute_classifier.joblib"
)

BASE = ["hr_mean", "hr_min", "hr_max", "rr_std", "rmssd", "breath_rate_est"]
HR_ONLY = ["hr_mean", "hr_min", "hr_max"]


def add_causal_rolling(df):
    """과거 5분 이동 평균/표준편차 (현재 분 제외 X: 현재 포함, 미래 제외)."""
    df = df.sort_values(["record", "minute"]).copy()
    for col in ["hr_mean", "rmssd", "breath_rate_est"]:
        g = df.groupby("record")[col]
        df[f"{col}_roll_mean_5"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
        df[f"{col}_roll_std_5"] = g.transform(lambda s: s.rolling(5, min_periods=2).std()).fillna(0)
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_causal_rolling(df)
    roll_cols = [c for c in df.columns if "_roll_" in c]
    full_feats = BASE + roll_cols
    hr_feats = HR_ONLY + [c for c in roll_cols if c.startswith("hr_mean")]

    print(f"전체 {len(df)}분, 기록 {df['record'].nunique()}개, 무호흡 비율 {df['label'].mean():.2%}\n")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(splitter.split(df, groups=df["record"]))
    train_df, test_df = df.iloc[tr], df.iloc[te]
    print(f"Train: {len(train_df)}분 / {train_df['record'].nunique()}기록")
    print(f"Test:  {len(test_df)}분 / {test_df['record'].nunique()}기록\n")

    results = {}
    for name, feats in [("심박수만(mmWave heart_rate 대응)", hr_feats),
                        ("심박+심박변이도+호흡추정(전체)", full_feats)]:
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        clf.fit(train_df[feats], train_df["label"])
        proba = clf.predict_proba(test_df[feats])[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(test_df["label"], proba)
        f1 = f1_score(test_df["label"], pred)
        results[name] = (auc, f1, clf, feats)

        print(f"=== {name} ===")
        print(f"AUC: {auc:.3f} | 무호흡 F1: {f1:.3f}")
        print(classification_report(test_df["label"], pred, target_names=["정상", "무호흡"], digits=3))
        print("혼동행렬 [정상, 무호흡]:")
        print(confusion_matrix(test_df["label"], pred))
        print()

    print("피처 중요도 (전체 모델):")
    full_clf = results["심박+심박변이도+호흡추정(전체)"][2]
    for n, imp in sorted(zip(full_feats, full_clf.feature_importances_), key=lambda x: -x[1])[:8]:
        print(f"  {n}: {imp:.3f}")

    # mmWave 배포용(심박수만)과 연구용(전체 피처) 모델을 따로 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "hr_only": {"model": results["심박수만(mmWave heart_rate 대응)"][2], "features": hr_feats},
        "full": {"model": full_clf, "features": full_feats},
    }, MODEL_PATH)
    print(f"\n모델 저장: {MODEL_PATH} (hr_only: mmWave 배포 가능 / full: ECG 파생 피처 필요)")


if __name__ == "__main__":
    main()
