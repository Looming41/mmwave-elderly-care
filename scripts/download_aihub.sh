#!/bin/bash
# AI Hub 데이터셋 71803(독거노인 돌봄용 위험감지 데이터) 다운로드
#
# 사전 준비:
#   1) https://aihub.or.kr 가입
#   2) 데이터셋 71803 페이지에서 이용신청(활용동의) 후 승인
#   3) 마이페이지에서 API 키 발급
#   4) export AIHUB_APIKEY="발급받은키"  (또는 아래 -aihubapikey 로 직접 전달)
set -e

cd "$(dirname "$0")/../data/raw"

if [ -z "$AIHUB_APIKEY" ]; then
  echo "AIHUB_APIKEY 환경변수가 설정되어 있지 않습니다."
  echo "export AIHUB_APIKEY=\"발급받은키\" 를 먼저 실행하세요."
  exit 1
fi

# 생체신호 원천/라벨링 파일만 선택 다운로드 (신규수집 30가구 기준)
# filekey는 -mode l 71803 결과에서 확인한 실제 파일 번호로 교체 필요
echo "필요한 filekey는 다음 명령으로 다시 확인하세요:"
echo "  aihubshell -mode l 71803"
echo
echo "파일 전체를 받으려면:"
echo "  aihubshell -mode d -datasetkey 71803"
echo
echo "생체신호 파일만 받으려면 (filekey 예시, 실제 번호 확인 후 수정):"
echo "  aihubshell -mode d -datasetkey 71803 -filekey 551756,551790"
