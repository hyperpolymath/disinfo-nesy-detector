# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2024 Hyperpolymath

FROM rust:1.85-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/nsai-detector /usr/local/bin/nsai-detector

# Metrics endpoint
EXPOSE 9090

ENTRYPOINT ["nsai-detector"]
CMD ["--help"]
