# PLIP-rs: Programming Language Internal Probing
# Multi-target Dockerfile for RTX 5060 Ti (local) and H100 (RIKEN)
#
# Build for RTX 5060 Ti:  docker build --build-arg TARGET=local -t plip-rs:local .
# Build for H100:         docker build --build-arg TARGET=h100 -t plip-rs:h100 .

ARG TARGET=local

# =============================================================================
# Stage 1: Rust builder
# =============================================================================
FROM rust:1.87-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first (for caching)
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs for dependency caching
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies only (cached layer)
RUN cargo build --release --no-default-features && rm -rf src

# Copy actual source
COPY src ./src

# Build the actual binary (no CUDA in builder - just the Rust code)
RUN touch src/main.rs && cargo build --release --no-default-features

# =============================================================================
# Stage 2: Runtime - Base selection
# =============================================================================
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04 AS base-local
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04 AS base-h100

# =============================================================================
# Stage 3: Runtime
# =============================================================================
FROM base-${TARGET} AS runtime

ARG TARGET=local

ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_LOG=info
ENV HF_HOME=/data/hf_cache
ENV PLIP_TARGET=${TARGET}

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 plip
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/plip-rs /app/plip-rs

# Copy corpus samples
COPY corpus /app/corpus

# Create data directories
RUN mkdir -p /data/corpus /data/outputs /data/hf_cache \
    && chown -R plip:plip /app /data

USER plip

# Default model based on target
ENV PLIP_DEFAULT_MODEL=${TARGET:+bigcode/starcoder2-3b}
ENV PLIP_DEFAULT_MODEL=${PLIP_DEFAULT_MODEL:-bigcode/starcoder2-3b}

# H100 uses larger model
RUN if [ "$TARGET" = "h100" ]; then \
        echo "PLIP_DEFAULT_MODEL=codellama/CodeLlama-34b-Instruct-hf" > /app/.env; \
    else \
        echo "PLIP_DEFAULT_MODEL=bigcode/starcoder2-3b" > /app/.env; \
    fi

ENTRYPOINT ["/app/plip-rs"]
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="PLIP-rs"
LABEL org.opencontainers.image.description="Programming Language Internal Probing in Rust"
LABEL org.opencontainers.image.source="https://github.com/PCfVW/plip-rs"
LABEL org.opencontainers.image.target="${TARGET}"
