#  llm-server.Dockerfile  -  Multi-Stage Build

# --- Build Stage ---
FROM debian:bullseye AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    ca-certificates

ARG LLAMA_CPP_REF=b4376
RUN git clone --branch "${LLAMA_CPP_REF}" --depth 1 https://github.com/ggerganov/llama.cpp.git /llama.cpp && \
    git -C /llama.cpp checkout --detach "${LLAMA_CPP_REF}"
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
