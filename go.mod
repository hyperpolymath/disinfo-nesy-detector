module gitlab.com/hyperpolymath/disinfo-nsai-detector

go 1.25.1

replace gitlab.com/hyperpolymath/disinfo-nsai-detector/pkg/onnx_wrapper => ./pkg/onnx_wrapper

replace gitlab.com/hyperpolymath/disinfo-nsai-detector/pkg/souffle_wrapper => ./pkg/souffle_wrapper

require (
	github.com/nats-io/nats.go v1.46.1
	github.com/prometheus/client_golang v1.23.2
	google.golang.org/protobuf v1.36.10
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/klauspost/compress v1.18.0 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/nats-io/nkeys v0.4.11 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.66.1 // indirect
	github.com/prometheus/procfs v0.16.1 // indirect
	go.yaml.in/yaml/v2 v2.4.2 // indirect
	golang.org/x/crypto v0.45.0 // indirect
	golang.org/x/sys v0.38.0 // indirect
)
