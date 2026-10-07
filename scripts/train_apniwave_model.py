"""
[연구용 전용 — mmWave 배포 불가] APNIWAVE(UWB 레이더 5구간 원시 진폭 신호)로
정상호흡/무호흡/기타이벤트 3분류 시도.

왜 "연구용 전용"인가: 이 데이터는 레이더의 5개 거리구간별 원시 진폭 신호
(Dist1~Dist5)다. 우리 MR60BHA2는 이런 원시 다중거리 신호를 UART로 안 주고
이미 계산된 심박수/호흡수/거리(단일값)만 주기 때문에, 여기서 학습한 모델은
우리 센서로 재현/배포할 수 없다. 레이더 기반 무호흡 분류가 어느 정도
가능한지 확인하는 순수 실험.

데이터 구조: 1,011개 구간(균형: 정상/무호흡/기타 각 337개), 구간당 170 타임스텝
x 5채널. 환자 ID 컬럼이 없어서 사람 단위 train/test 분리가 불가능하다 —
무작위 분리라 인접 구간이 train/test에 섞일 수 있어 정확도가 부풀려질 수 있음.

사용법:
  python3 scripts/train_apniwave_model.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.expanduser("~/다운로드/APNIWAVE_extracted")
WINDOW = 170
N_CHANNELS = 5


def extract_features(window_df):
    feats = {}
    for col in window_df.columns:
        sig = window_df[col].values
        feats[f"{col}_mean"] = sig.mean()
        feats[f"{col}_std"] = sig.std()
        feats[f"{col}_min"] = sig.min()
        feats[f"{col}_max"] = sig.max()
        feats[f"{col}_range"] = sig.max() - sig.min()
        # 변화량(움직임/호흡 추정 proxy)
        feats[f"{col}_meandiff_abs"] = np.abs(np.diff(sig)).mean()
    return feats


def main():
    labels = pd.read_csv(os.path.join(DATA_DIR, "Labels Three Events.csv"))["Labels"].values
    signal = pd.read_csv(os.path.join(DATA_DIR, "Multiple Distances Three Events.csv"))

    n_windows = len(labels)
    assert len(signal) == n_windows * WINDOW, (
        f"신호 행수({len(signal)})가 라벨수({n_windows})*{WINDOW}와 안 맞음"
    )

    rows = []
    for i in range(n_windows):
        seg = signal.iloc[i * WINDOW:(i + 1) * WINDOW]
        feats = extract_features(seg)
        feats["label"] = labels[i]
        rows.append(feats)

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c != "label"]

    print(f"전체 {len(df)}개 구간, 라벨 분포: {df['label'].value_counts().to_dict()}")
    print("⚠ 환자 ID가 없어 무작위 분리 (사람 단위 분리 불가 — 결과를 곧이곧대로 믿지 말 것)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df[feat_cols], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("=== 분류 리포트 (0=정상호흡, 1=무호흡, 2=기타이벤트) ===")
    print(classification_report(y_test, pred, digits=3))
    print("혼동행렬:")
    print(confusion_matrix(y_test, pred))

    print("\n피처 중요도 (상위 10개):")
    for n, imp in sorted(zip(feat_cols, clf.feature_importances_), key=lambda x: -x[1])[:10]:
        print(f"  {n}: {imp:.3f}")


if __name__ == "__main__":
    main()
