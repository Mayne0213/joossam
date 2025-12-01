#!/bin/bash

# Joossam Docker Cleanup Script
# 공통 유틸리티 함수 로드
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# 스크립트 설정
setup_script

log_info "🧹 Joossam Docker 리소스 정리 시작..."

# 확인 메시지
if ! confirm_action "모든 Joossam 관련 Docker 리소스를 정리하시겠습니까?"; then
    log_info "정리를 취소했습니다."
    exit 0
fi

# Docker 리소스 정리 실행
docker_cleanup_joossam

log_info "✅ Joossam Docker 리소스 정리 완료!"
