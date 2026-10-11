//go:build !(linux && cgo)

// Package rk is the cgo binding to librkllmrt.so. This stub lets the rest of
// the tree build and test on machines without the runtime (e.g. macOS).
package rk

import "rkllm-go/internal/engine"

type stub struct{}

// New returns the engine for this build.
func New() engine.Engine { return stub{} }

func (stub) Load(engine.Params) (engine.Handle, error) { return nil, engine.ErrUnsupported }
func (stub) Version() string                           { return "none" }
func (stub) PerRunMaxTokens() bool                     { return false }
