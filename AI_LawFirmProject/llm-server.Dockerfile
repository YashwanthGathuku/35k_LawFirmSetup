#  llm-server.Dockerfile  -  Multi-Stage Build

# --- Build Stage ---
FROM debian:bullseye AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch b4096 https://github.com/ggerganov/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp

RUN mkdir build && cd build && \
    cmake .. -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF && \
    cmake --build .

# --- Runtime Stage ---
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

RUN mkdir /models
WORKDIR /
EXPOSE 8080

ENTRYPOINT ["llama-server"]
