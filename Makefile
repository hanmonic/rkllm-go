# rkllm-go v2 build. Runs natively on the arm64 board or container (needs both runtimes installed).
#   make            -> rkllm-server-131 (runtime 1.3.1) and rkllm-server-123 (runtime 1.2.3)
#   make test       -> pure-Go unit tests (no runtime needed; works on macOS)

.PHONY: all test clean headers
all: headers rkllm-server-131 rkllm-server-123

# Rockchip's headers are fetched, not vendored (see NOTICE)
headers:
	@[ -f third_party/rkllm/1.3.1/rkllm.h ] && [ -f third_party/rkllm/1.2.3/rkllm.h ] || scripts/fetch-headers.sh

rkllm-server-131: headers FORCE
	go build -tags rk131 -o $@ ./cmd/server

rkllm-server-123: headers FORCE
	go build -tags rk123 -o $@ ./cmd/server

rkllm-memtest: FORCE
	go build -o $@ ./cmd/rkllm-memtest

test:
	go test ./internal/...

clean:
	rm -f rkllm-server-131 rkllm-server-123 rkllm-memtest

FORCE:
