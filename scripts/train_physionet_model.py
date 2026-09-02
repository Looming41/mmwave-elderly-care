"""
PhysioNet 2019로 "심박+호흡 → 생리학적 위험(패혈증 6시간 전조)" 모델을 대규모로 학습.

71803(27명)보다 훨씬 큰 표본(2만여 명)으로 "심박/호흡 이상 패턴 → 위험" 관계를
더 안정적으로 학습시키는 게 목적. 이 모델의 위험확률 출력을 나중에 71803 모델의
추가 피처로 넣어서(스태킹) 성능이 오르는지 확인한다.

patient_id 단위로 train/test 분리.
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "physionet2019_labeled.csv"
)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "physionet_risk_classifier.joblib"
)

BASE_COLS = ["heart_rate", "breath_rate"]
WINDOWS = [3, 6]


def add_rolling_features(group):
    group = group.sort_values("hours_since_admit")
    for col in BASE_COLS:
        group[f"{col}_diff"] = group[col].diff()
        for w in WINDOWS:
            group[f"{col}_roll_mean_{w}"] = group[col].rolling(w, min_periods=1).mean()
            group[f"{col}_roll_std_{w}"] = group[col].rolling(w, min_periods=2).std()
    return group


FEATURES = [
    "heart_rate", "breath_rate",
    "heart_rate_diff", "heart_rate_roll_mean_3", "heart_rate_roll_std_3",
    "heart_rate_roll_mean_6", "heart_rate_roll_std_6",
    "breath_rate_diff", "breath_rate_roll_mean_3", "breath_rate_roll_std_3",
    "breath_rate_roll_mean_6", "breath_rate_roll_std_6",
]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.groupby("patient_id", group_keys=False)[df.columns].apply(add_rolling_features)

    std_cols = [c for c in df.columns if c.endswith("_std_3") or c.endswith("_std_6")]
    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    df[std_cols] = df[std_cols].fillna(0)
    df[diff_cols] = df[diff_cols].fillna(0)

    print(f"전체 {len(df)}행, 환자 {df['patient_id'].nunique()}명")
    print(f"양성 비율: {df['SepsisLabel'].mean():.2%}\n")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    X_train, y_train = train_df[FEATURES], train_df["SepsisLabel"]
    X_test, y_test = test_df[FEATURES], test_df["SepsisLabel"]

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("=== 분류 리포트 ===")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")

    print("\n피처 중요도:")
    for name, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
