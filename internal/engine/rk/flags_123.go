//go:build linux && cgo && rk123

package rk

/*
#cgo CFLAGS: -I${SRCDIR}/compat -I${SRCDIR}/../../../third_party/rkllm/1.2.3
#cgo LDFLAGS: -L/opt/rkllm/runtime -lrkllmrt -Wl,-rpath,/opt/rkllm/runtime
*/
import "C"

const runtimeVersion = "1.2.3"
