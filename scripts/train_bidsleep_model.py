"""
BIDSleep(EEG 골드스탠다드 라벨)로 심박수만 사용한 수면단계 분류.

mmWave가 밤새 심박수만 기록한다면 얻을 수 있는 피처: hr_mean, hr_std, hr_min,
hr_max, hr_diff, n_samples (motion.csv 가속도계는 mmWave에 없으니 미사용).

subject 단위로 train/test 분리 (같은 사람의 여러 밤이 섞이면 leakage).
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "bidsleep_labeled.csv"
)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "bidsleep_stage_classifier.joblib"
)

FEATURES = ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_diff", "n_samples"]


def main():
    df = pd.read_csv(DATA_PATH)

    print(f"전체 {len(df)}행, 대상자 {df['subject'].nunique()}명\n")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["subject"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    print(f"Train: {len(train_df)}행 / {train_df['subject'].nunique()}명")
    print(f"Test:  {len(test_df)}행 / {test_df['subject'].nunique()}명\n")

    X_train, y_train = train_df[FEATURES], train_df["stage"]
    X_test, y_test = test_df[FEATURES], test_df["stage"]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== 분류 리포트 (Test, EEG 골드스탠다드 기준) ===")
    print(classification_report(y_test, y_pred, digits=3))

    labels = sorted(y_test.unique())
    print(f"혼동행렬 {labels}:")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    print("\n피처 중요도:")
    for name, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
