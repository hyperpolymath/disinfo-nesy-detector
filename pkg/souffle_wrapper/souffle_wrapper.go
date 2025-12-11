package souffle_wrapper

import (
	"context"
)

type NeuralFeatures map[string]float32
type DgraphFacts map[string]string

func RunDatalog(ctx context.Context, neuralFeatures NeuralFeatures, dgraphFacts DgraphFacts) (string, string, error) {
	return "SAFE", "No rules fired (placeholder)", nil
}
