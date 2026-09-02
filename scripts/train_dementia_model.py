"""
AI Hub 226(치매 고위험군 웨어러블 라이프로그) - 장기(person-level) 치매위험군 분류.

이전 버전의 문제: 같은 사람의 여러 밤(person-night)을 각각 독립 샘플처럼 넣었음.
같은 사람은 어느 밤이든 진단(DIAG_NM)이 동일하므로, 실제 통계적 샘플 수는
"밤 개수"가 아니라 "사람 수"다. 밤 단위로 쪼개면 같은 신호를 여러 번 세는
pseudo-replication 오류가 생기고, train/val이 우연히 person-disjoint가 아니면
leakage도 생긴다(이번엔 AI Hub가 이미 person-disjoint로 나눠놔서 leakage는 없었지만
샘플 뻥튀기 문제는 여전히 있었음).

수정: 사람(EMAIL) 단위로 여러 밤을 집계(평균/표준편차)해서 "장기 패턴"을 피처로 삼는다.
mmWave를 몇 주~몇 달 장기간 돌린다면 얻을 수 있는 값과 성격이 같다.
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "226_라이프로그",
    "128.치매_고위험군_라이프로그", "01.데이터",
)

RAW_FEATURES = ["sleep_hr_average", "sleep_hr_lowest", "sleep_rmssd"]


def load_person_level(sleep_path, label_path):
    sleep = pd.read_csv(sleep_path)
    label = pd.read_csv(label_path)
    df = sleep.merge(label, left_on="EMAIL", right_on="SAMPLE_EMAIL", how="inner")
    df = df.dropna(subset=RAW_FEATURES + ["DIAG_NM"])

    agg = df.groupby("EMAIL").agg(
        n_nights=("DIAG_NM", "size"),
        hr_avg_mean=("sleep_hr_average", "mean"),
        hr_avg_std=("sleep_hr_average", "std"),
        hr_lowest_mean=("sleep_hr_lowest", "mean"),
        hr_lowest_std=("sleep_hr_lowest", "std"),
        rmssd_mean=("sleep_rmssd", "mean"),
        rmssd_std=("sleep_rmssd", "std"),
        DIAG_NM=("DIAG_NM", "first"),
    ).reset_index()

    agg = agg.fillna(0)  # 밤이 1개뿐이라 std가 NaN인 경우
    return agg


def main():
    train = load_person_level(
        os.path.join(BASE, "1.Training/원천데이터/2.수면/train_sleep.csv"),
        os.path.join(BASE, "1.Training/라벨링데이터/2.수면/training_label.csv"),
    )
    val = load_person_level(
        os.path.join(BASE, "2.Validation/원천데이터/2.수면/val_sleep.csv"),
        os.path.join(BASE, "2.Validation/라벨링데이터/2.수면/val_label.csv"),
    )

    features = [
        "n_nights", "hr_avg_mean", "hr_avg_std",
        "hr_lowest_mean", "hr_lowest_std", "rmssd_mean", "rmssd_std",
    ]

    print(f"Train: {len(train)}명, Val: {len(val)}명")
    print("\nTrain DIAG_NM 분포:")
    print(train["DIAG_NM"].value_counts())

    X_train, y_train = train[features], train["DIAG_NM"]
    X_val, y_val = val[features], val["DIAG_NM"]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=5, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print("\n=== 분류 리포트 (person 단위, Validation) ===")
    print(classification_report(y_val, y_pred, digits=3))
    print("혼동행렬 (행:실제, 열:예측, 알파벳순 CN/Dem/MCI):")
    print(confusion_matrix(y_val, y_pred))

    print("\n피처 중요도:")
    for name, imp in sorted(
        zip(features, clf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name}: {imp:.3f}")


if __name__ == "__main__":
    main()
