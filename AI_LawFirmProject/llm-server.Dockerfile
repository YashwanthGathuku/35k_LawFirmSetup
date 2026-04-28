#  llm-server.Dockerfile  -  Multi-Stage  Build

#  =============================================================
#  Stage  1:  Build  llama.cpp  from  a  pinned  release  tag
#  =============================================================
FROM debian:bookworm-slim AS builder

#  Install  build  dependencies  and  clean  up  apt  lists
RUN apt-get  update  &&  apt-get  install  -y  --no-install-recommends  \
    git  \
    cmake  \
    build-essential  \
    ca-certificates  \
    &&  rm  -rf  /var/lib/apt/lists/*

#  Shallow-clone  the  pinned  tag  for  reproducible  builds.
#  llama.cpp  uses  "b<build_number>"  style  tags  (e.g.  b4096  =  build  #4096).
#  Update  LLAMACPP_TAG  intentionally  when  upgrading.
ARG LLAMACPP_TAG=b4096
RUN git  clone  --depth  1  --branch  ${LLAMACPP_TAG}  \
    https://github.com/ggerganov/llama.cpp.git  /llama.cpp

WORKDIR /llama.cpp

#  Build  a  static  binary  so  the  runtime  stage  needs  no  shared  libs
RUN mkdir  build  &&  cd  build  &&  \
    cmake  ..  -DLLAMA_CURL=OFF  -DBUILD_SHARED_LIBS=OFF  &&  \
    cmake  --build  .  --config  Release

#  =============================================================
#  Stage  2:  Minimal  runtime  image  (no  build  tools)
#  =============================================================
FROM debian:bookworm-slim

#  Only  the  runtime  libraries  needed  to  run  the  binary
RUN apt-get  update  &&  apt-get  install  -y  --no-install-recommends  \
    libgomp1  \
    ca-certificates  \
    &&  rm  -rf  /var/lib/apt/lists/*

COPY --from=builder  /llama.cpp/build/bin/llama-server  /usr/local/bin/llama-server

RUN mkdir  /models
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llama-server"]
