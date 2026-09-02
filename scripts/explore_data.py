"""
data/raw/ 아래 압축을 풀어둔 AI Hub 71803(생체신호) CSV들을 훑어서
실제 컬럼명, 샘플링 주기, 값 범위를 확인하는 스크립트.

사용법:
  1) aihubshell로 받은 zip을 data/raw/ 에서 압축 해제
     예: unzip "TS_A.신규수집_02.생체신호.zip" -d data/raw/biosignal
  2) python3 scripts/explore_data.py
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

HR_KEYWORDS = ["hr", "heart", "심박", "pulse", "맥박"]


def find_csvs(root):
    return glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)


def main():
    csvs = find_csvs(RAW_DIR)
    if not csvs:
        print(f"{RAW_DIR} 아래에서 CSV를 찾지 못했습니다.")
        print("먼저 aihubshell로 받은 zip을 data/raw/ 에 압축 해제하세요.")
        return

    print(f"CSV {len(csvs)}개 발견\n")

    for path in csvs[:20]:
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception as e:
            print(f"[읽기 실패] {path}: {e}")
            continue

        hr_cols = [
            c for c in df.columns
            if any(k in c.lower() or k in c for k in HR_KEYWORDS)
        ]

        print(f"--- {os.path.relpath(path, RAW_DIR)} ---")
        print(f"  컬럼: {list(df.columns)}")
        if hr_cols:
            print(f"  심박수 관련 컬럼 후보: {hr_cols}")
        print()


if __name__ == "__main__":
    main()
