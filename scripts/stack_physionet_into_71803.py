"""
PhysioNet 2019로 학습한 대규모 위험예측 모델(physionet_risk_classifier)의 예측확률을
71803 이상감지 모델의 추가 피처로 넣어서(스태킹) 성능이 오르는지 검증한다.

베이스라인(71803 단독, anomaly=주의+위험): F1 0.505
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "aihub_71803_features.csv"
)
PHYSIONET_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "physionet_risk_classifier.joblib"
)

PHYSIONET_FEATURES = [
    "heart_rate", "breath_rate",
    "heart_rate_diff", "heart_rate_roll_mean_3", "heart_rate_roll_std_3",
    "heart_rate_roll_mean_6", "heart_rate_roll_std_6",
    "breath_rate_diff", "breath_rate_roll_mean_3", "breath_rate_roll_std_3",
    "breath_rate_roll_mean_6", "breath_rate_roll_std_6",
]


def main():
    df = pd.read_csv(DATA_PATH)
    df["is_anomaly"] = df["label"].isin(["주의", "위험"]).astype(int)

    # PhysioNet 모델로 "생리학적 위험확률" 피처 생성
    physionet_clf = joblib.load(PHYSIONET_MODEL_PATH)
    df["physionet_risk_score"] = physionet_clf.predict_proba(df[PHYSIONET_FEATURES])[:, 1]

    features = PHYSIONET_FEATURES + ["physionet_risk_score"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["subject_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    X_train, y_train = train_df[features], train_df["is_anomaly"]
    X_test, y_test = test_df[features], test_df["is_anomaly"]

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== 스태킹 모델 (71803 피처 + physionet_risk_score) ===")
    print(classification_report(y_test, y_pred, digits=3))
    f1 = f1_score(y_test, y_pred)
    print(f"F1: {f1:.3f}  (베이스라인 71803 단독: 0.505)")

    print("\n피처 중요도:")
    for name, imp in sorted(zip(features, clf.feature_importances_), key=lambda x: -x[1]):
        marker = " <- 스태킹 추가" if name == "physionet_risk_score" else ""
        print(f"  {name}: {imp:.3f}{marker}")


if __name__ == "__main__":
    main()
