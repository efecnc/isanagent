# ---------- builder ----------
    FROM rust:1.90-bookworm AS builder

    WORKDIR /app
    
    # install deps often needed by crates (adjust if not needed)
    RUN apt-get update && apt-get install -y \
        pkg-config \
        libssl-dev \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*
    
    # cache dependencies
    COPY Cargo.toml Cargo.lock ./
    RUN cargo fetch
    
    # build actual project
    COPY . .
    RUN cargo build --release
    
    # ---------- runtime ----------
    FROM debian:bookworm-slim AS runtime
    
    # minimal runtime deps (openssl + certs commonly needed)
    RUN apt-get update && apt-get install -y \
        ca-certificates \
        libssl3 \
        && rm -rf /var/lib/apt/lists/*
    
    # copy binary (world-readable, on PATH)
    COPY --from=builder /app/target/release/isanagent /usr/local/bin/isanagent

    # Run as a non-root user. The agent executes model-authored shell/Python, so running the
    # container process as root needlessly widens the blast radius of a runaway command or a
    # container escape. `-m` creates a home dir so the default workspace (`~/.isanagent`) and the
    # first-run onboarding are writable without extra mounts. Override with `--build-arg UID=...`.
    ARG UID=1000
    RUN useradd -u ${UID} -m -s /bin/bash appuser
    # Pin HOME explicitly: the default workspace resolves via `~/.isanagent` (shellexpand reads
    # $HOME), so make the contract robust even if a caller overrides the entrypoint.
    ENV HOME=/home/appuser
    USER appuser
    WORKDIR /home/appuser

    ENTRYPOINT ["isanagent"]