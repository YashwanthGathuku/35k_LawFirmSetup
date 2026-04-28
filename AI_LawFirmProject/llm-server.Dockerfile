#  llm-server.Dockerfile  -  Final  Single-Stage  Build

FROM debian:bullseye

#  1.  Install  all  build  and  runtime  dependencies  in  one  go
RUN apt-get  update  &&  apt-get  install  -y  --no-install-recommends  \
    git  \
    cmake  \
    build-essential  \
    ca-certificates  \
    libgomp1

#  2.  Clone  the  llama.cpp  repository
RUN git  clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /llama.cpp

#  3.  Build  the  code.  The  BUILD_SHARED_LIBS=OFF  flag  is  crucial.
RUN mkdir  build  &&  cd build  &&  \
    cmake  ..  -DLLAMA_CURL=OFF  -DBUILD_SHARED_LIBS=OFF  &&  \
    cmake  --build  .

#  4.  Set  up  the  runtime  environment
WORKDIR /
RUN mkdir  /models
EXPOSE 8080

#  5.  Define  the  program  to  run  when  the  container  starts
ENTRYPOINT [  "/llama.cpp/build/bin/llama-server" ]
