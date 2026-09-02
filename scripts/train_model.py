"""
AI Hub 71803 데이터로 이상 상황 감지 모델을 학습한다.

설계 결정:
  - 피처는 mmWave(MR60BHA2)가 실시간으로 낼 수 있는 heart_rate, breath_rate
    (+파생 롤링 피처)만 사용한다. spo2/sleep_phase/steps 등 AI Hub 전용 값은
    실배포 시 mmWave 센서에 없으므로 학습에서 제외한다.
  - 두 가지 타겟 정의를 비교한다:
      risk_only : label == "위험"           (좁은 정의, 표본 3.4%)
      anomaly   : label in ("주의", "위험")  (넓은 정의, 표본 ~22%)
    heart_rate/breath_rate 두 신호만으론 "위험"만 좁게 잡기보다
    "이상 신호 전반"을 잡는 쪽이 표본도 많고 신호도 더 잘 맞을 거라는 가설 검증.
  - subject_id 단위로 train/test를 나눈다 (data leakage 방지).

사용법:
  python3 scripts/train_model.py
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "aihub_71803_features.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURES = [
    "heart_rate", "breath_rate",
    "heart_rate_diff", "heart_rate_roll_mean_3", "heart_rate_roll_std_3",
    "heart_rate_roll_mean_6", "heart_rate_roll_std_6",
    "breath_rate_diff", "breath_rate_roll_mean_3", "breath_rate_roll_std_3",
    "breath_rate_roll_mean_6", "breath_rate_roll_std_6",
]

TARGETS = {
    "risk_only": lambda df: (df["label"] == "위험").astype(int),
    "anomaly": lambda df: df["label"].isin(["주의", "위험"]).astype(int),
}


def train_one(df, target_name, target_fn, train_idx, test_idx):
    df = df.copy()
    df["target"] = target_fn(df)

    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    X_train, y_train = train_df[FEATURES], train_df["target"]
    X_test, y_test = test_df[FEATURES], test_df["target"]

    print(f"\n{'='*50}")
    print(f"타겟: {target_name}  (양성 비율: train {y_train.mean():.2%} / test {y_test.mean():.2%})")
    print(f"{'='*50}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, digits=3))
    print("혼동행렬:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(MODEL_DIR, f"{target_name}_classifier.joblib")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"모델 저장: {model_path}")

    return f1_score(y_test, y_pred, pos_label=1)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"전체 {len(df)}행, 대상자 {df['subject_id'].nunique()}명")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["subject_id"]))

    results = {}
    for name, fn in TARGETS.items():
        results[name] = train_one(df, name, fn, train_idx, test_idx)

    print(f"\n{'='*50}")
    print("F1 비교")
    for name, f1 in results.items():
        print(f"  {name}: {f1:.3f}")


if __name__ == "__main__":
    main()
