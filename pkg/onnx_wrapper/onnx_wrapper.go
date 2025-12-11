package onnx_wrapper

import (
	"context"
)

func InitRuntime() error {
	return nil
}

func ShutdownRuntime() {}

func RunInference(ctx context.Context, contentHash string) (map[string]float32, error) {
	return map[string]float32{"fakeness_score": 0.5, "emotion_score": 0.3}, nil
}
