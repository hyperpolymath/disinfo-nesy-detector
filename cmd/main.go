package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"reflect"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/protobuf/proto"
	"gitlab.com/hyperpolymath/disinfo-nsai-detector/pkg/model_pb"
	"gitlab.com/hyperpolymath/disinfo-nsai-detector/pkg/onnx_wrapper"
	"gitlab.com/hyperpolymath/disinfo-nsai-detector/pkg/souffle_wrapper"
)

const (
	NATS_URL      = "nats://nats:4222"
	STREAM_NAME   = "INFERENCE_JOBS"
	SUBJECT_INPUT = "disinfo.raw"
	SUBJECT_DLQ   = "disinfo.dlq"
	CONSUMER_NAME = "detector_worker"
	METRICS_PORT  = "9090"
)

var (
	metrics = struct {
		messagesProcessed prometheus.Counter
		errors            prometheus.Counter
		latency           prometheus.Histogram
	}{
		messagesProcessed: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "nsai_messages_processed_total",
			Help: "Total number of messages processed",
		}),
		errors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "nsai_errors_total",
			Help: "Total number of errors",
		}),
		latency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "nsai_processing_latency_seconds",
			Help:    "Latency of message processing",
			Buckets: prometheus.DefBuckets,
		}),
	}
)

func init() {
	prometheus.MustRegister(metrics.messagesProcessed, metrics.errors, metrics.latency)
	log.Println("🔍 Introspecting jetstream.Msg type...")
	msgType := reflect.TypeOf((*jetstream.Msg)(nil)).Elem()
	for i := 0; i < msgType.NumMethod(); i++ {
		method := msgType.Method(i)
		log.Printf("  Method: %s, Type: %s", method.Name, method.Type)
	}
}

func main() {
	log.Println("🚀 Starting NSAI Detector Service (Golden Repo Edition)")

	// Initialize ONNX runtime
	if err := onnx_wrapper.InitRuntime(); err != nil {
		log.Fatalf("❌ Failed to initialize ONNX runtime: %v", err)
	}
	defer onnx_wrapper.ShutdownRuntime()

	// Connect to NATS
	nc, err := nats.Connect(NATS_URL, nats.ErrorHandler(func(_ *nats.Conn, _ *nats.Subscription, err error) {
		log.Printf("⚠️ NATS error: %v", err)
		metrics.errors.Inc()
	}))
	if err != nil {
		log.Fatalf("❌ Could not connect to NATS: %v", err)
	}
	defer nc.Close()

	// Get JetStream context
	js, err := jetstream.New(nc)
	if err != nil {
		log.Fatalf("❌ Could not get JetStream context: %v", err)
	}

	// Create or get the stream
	ctx := context.Background()
	_, err = js.CreateStream(ctx, jetstream.StreamConfig{
		Name:     STREAM_NAME,
		Subjects: []string{SUBJECT_INPUT},
	})
	if err != nil {
		log.Fatalf("❌ Could not create stream: %v", err)
	}

	// Start metrics server
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		log.Printf("📊 Metrics server running on :%s", METRICS_PORT)
		if err := http.ListenAndServe(":"+METRICS_PORT, nil); err != nil {
			log.Fatalf("❌ Metrics server failed: %v", err)
		}
	}()

	// Create a pull consumer
	_, err = js.CreateOrUpdateConsumer(ctx, STREAM_NAME, jetstream.ConsumerConfig{
		Durable:       CONSUMER_NAME,
		AckPolicy:     jetstream.AckExplicitPolicy,
		DeliverPolicy: jetstream.DeliverAllPolicy,
	})
	if err != nil {
		log.Fatalf("❌ Could not create consumer: %v", err)
	}

	// Start consumer with context for graceful shutdown
	sigCtx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Get the consumer
	consumer, err := js.Consumer(ctx, STREAM_NAME, CONSUMER_NAME)
	if err != nil {
		log.Fatalf("❌ Could not get consumer: %v", err)
	}

	// Fetch messages
	log.Printf("📩 Listening for messages on %s...", SUBJECT_INPUT)
	for {
		select {
		case <-sigCtx.Done():
			log.Println("🛑 Shutting down gracefully...")
			return
		default:
			msgs, err := consumer.Fetch(1, jetstream.FetchMaxWait(5*time.Second))
			if err != nil {
				if err == nats.ErrTimeout {
					continue
				}
				log.Printf("⚠️ Fetch error: %v", err)
				continue
			}
			for msg := range msgs.Messages() {
				// Pre-processing hook
				log.Printf("🔌 Pre-processing message: %v", msg.Subject())

				processMessage(sigCtx, msg)

				// Post-processing hook
				log.Printf("🔌 Post-processing message: %v", msg.Subject())
			}
		}
	}
}

func processMessage(ctx context.Context, msg jetstream.Msg) {
	// Aspect-oriented hook for acknowledgment
	defer msg.Ack()
	start := time.Now()
	defer func() { metrics.latency.Observe(time.Since(start).Seconds()) }()

	// Introspect the message type
	log.Printf("🔍 Message type: %T", msg)

	var input model_pb.AnalysisInput
	if err := proto.Unmarshal(msg.Data(), &input); err != nil {
		log.Printf("❌ Unmarshal error: %v", err)
		metrics.errors.Inc()
		return
	}
	metrics.messagesProcessed.Inc()

	// Neuro-Symbolic Pipeline
	neuralFeatures, err := onnx_wrapper.RunInference(ctx, input.ContentHash)
	if err != nil {
		log.Printf("❌ ONNX inference error: %v", err)
		metrics.errors.Inc()
		return
	}

	dgraphFacts := fetchDgraphFacts(ctx, input.SourceId)
	verdict, explanation, err := souffle_wrapper.RunDatalog(ctx, neuralFeatures, dgraphFacts)
	if err != nil {
		log.Printf("❌ Souffle error: %v", err)
		metrics.errors.Inc()
		return
	}

	log.Printf("✅ Verdict for %s: %s | %s", input.ContentHash, verdict, explanation)
}

func fetchDgraphFacts(ctx context.Context, sourceID string) map[string]string {
	return map[string]string{"source_trusted": "true"}
}
