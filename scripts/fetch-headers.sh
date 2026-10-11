#!/bin/sh
# Downloads Rockchip's RKLLM headers (unmodified) from https://github.com/airockchip/rknn-llm
# into third_party/rkllm/<version>/rkllm.h. They are not vendored here; see NOTICE.
set -eu
cd "$(dirname "$0")/.."
PATH_IN_REPO=rkllm-runtime/Linux/librkllm_api/include/rkllm.h
for v in "${@:-1.2.3 1.3.1}"; do
  for ver in $v; do
    mkdir -p "third_party/rkllm/$ver"
    curl -fsSL "https://raw.githubusercontent.com/airockchip/rknn-llm/release-v$ver/$PATH_IN_REPO" -o "third_party/rkllm/$ver/rkllm.h"
    echo "fetched third_party/rkllm/$ver/rkllm.h"
  done
done
