"""
APSAA(32명, 흉부신호 기반 호흡수) + Apnea-ECG(35명, ECG유도 호흡수)를 합쳐서
"1분 단위 무호흡/저호흡" 분류 모델을 mmWave 호환 피처만으로 학습한다.

두 데이터셋의 공통 피처만 사용: hr_mean, hr_min, hr_max, breath_rate_est
(+과거 5분 롤링 평균/표준편차, 실시간 배포와 동일하게 미래값 사용 안 함)

비교:
  1) APSAA 단독
  2) Apnea-ECG 단독 (지난번 결과, 재확인용)
  3) 둘을 합친 모델  <- 표본이 늘어나면 정확도가 오르는지 확인
사람(기록) 단위로 train/test 분리.
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "apnea_combined_classifier.joblib")

COMMON = ["hr_mean", "hr_min", "hr_max", "breath_rate_est"]


def load_apsaa():
    df = pd.read_csv(os.path.join(BASE_DIR, "apsaa_minutes.csv"))
    df["group"] = "apsaa_" + df["subject"]
    df["source"] = "apsaa"
    return df[["group", "source", "minute", "label"] + COMMON]


def load_apnea_ecg():
    df = pd.read_csv(os.path.join(BASE_DIR, "apnea_minutes.csv"))
    df = df.rename(columns={"record": "group"})
    df["group"] = "aecg_" + df["group"]
    df["source"] = "apnea_ecg"
    return df[["group", "source", "minute", "label"] + COMMON]


def add_causal_rolling(df):
    df = df.sort_values(["group", "minute"]).copy()
    for col in ["hr_mean", "breath_rate_est"]:
        g = df.groupby("group")[col]
        df[f"{col}_roll_mean_5"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
        df[f"{col}_roll_std_5"] = g.transform(lambda s: s.rolling(5, min_periods=2).std()).fillna(0)
    return df


def evaluate(name, df):
    df = add_causal_rolling(df)
    feats = COMMON + [c for c in df.columns if "_roll_" in c]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(splitter.split(df, groups=df["group"]))
    train_df, test_df = df.iloc[tr], df.iloc[te]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(train_df[feats], train_df["label"])
    proba = clf.predict_proba(test_df[feats])[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(test_df["label"], proba)
    f1 = f1_score(test_df["label"], pred)

    print(f"\n{'='*55}\n{name}  (train {train_df['group'].nunique()}명 / test {test_df['group'].nunique()}명, {len(df)}분)\n{'='*55}")
    print(f"AUC: {auc:.3f} | 무호흡 F1: {f1:.3f}")
    print(classification_report(test_df["label"], pred, target_names=["정상", "무호흡/저호흡"], digits=3))
    print("혼동행렬:")
    print(confusion_matrix(test_df["label"], pred))
    return auc, f1, clf, feats


def main():
    apsaa = load_apsaa()
    aecg = load_apnea_ecg()
    combined = pd.concat([apsaa, aecg], ignore_index=True)

    r1 = evaluate("1) APSAA 단독 (흉부신호 기반 호흡수)", apsaa)
    r2 = evaluate("2) Apnea-ECG 단독 (ECG유도 호흡수)", aecg)
    r3 = evaluate("3) 합친 데이터 (67명)", combined)

    print(f"\n{'='*55}\nAUC 비교: APSAA {r1[0]:.3f} | Apnea-ECG {r2[0]:.3f} | 합친 모델 {r3[0]:.3f}")
    print(f"F1 비교:  APSAA {r1[1]:.3f} | Apnea-ECG {r2[1]:.3f} | 합친 모델 {r3[1]:.3f}")

    print("\n합친 모델 피처 중요도:")
    for n, imp in sorted(zip(r3[3], r3[2].feature_importances_), key=lambda x: -x[1]):
        print(f"  {n}: {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": r3[2], "features": r3[3]}, MODEL_PATH)
    print(f"\n합친 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
