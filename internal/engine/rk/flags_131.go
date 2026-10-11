//go:build linux && cgo && !rk123

package rk

/*
#cgo CFLAGS: -I${SRCDIR}/compat -I${SRCDIR}/../../../third_party/rkllm/1.3.1 -DRKLLM_V131
#cgo LDFLAGS: -L/opt/rkllm-131/server/lib -lrkllmrt -Wl,-rpath,/opt/rkllm-131/server/lib
*/
import "C"

const runtimeVersion = "1.3.1"
