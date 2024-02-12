FROM rust:1.71 AS rust_base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS cuda_base

FROM cuda_base AS merged_base
COPY --from=rust_base /usr/local/cargo /usr/local/cargo
COPY --from=rust_base /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"
ENV DEBIAN_FRONTEND noninteractive

RUN rustup component add rustfmt
RUN rustup component add rust-analyzer

RUN apt-get update -y && apt-get install -y clang-format
RUN apt-get update -y && apt-get install -y clang-tools
RUN apt-get update -y && apt-get install -y clang
RUN apt-get update -y && apt-get install -y clangd
RUN apt-get update -y && apt-get install -y libc++-dev
RUN apt-get update -y && apt-get install -y libc++1
RUN apt-get update -y && apt-get install -y libc++abi-dev
RUN apt-get update -y && apt-get install -y libc++abi1
RUN apt-get update -y && apt-get install -y libclang-dev
RUN apt-get update -y && apt-get install -y libclang1
RUN apt-get update -y && apt-get install -y liblldb-dev
RUN apt-get update -y && apt-get install -y libomp-dev
RUN apt-get update -y && apt-get install -y libomp5
RUN apt-get update -y && apt-get install -y lld
RUN apt-get update -y && apt-get install -y lldb
RUN apt-get update -y && apt-get install -y llvm-dev
RUN apt-get update -y && apt-get install -y llvm-runtime
RUN apt-get update -y && apt-get install -y llvm
RUN apt-get update -y && apt-get install -y python3-clang


RUN apt-get update && \
    apt-get install -y lsb-release

RUN apt-get update && apt-get -y install cmake  

RUN apt-get update -y && apt-get install -y ripgrep 

RUN apt-get update -y && apt-get install -y gettext

RUN apt-get update && apt-get install -y python3.10 python3-pip

RUN apt-get update && apt-get install -y vim wget sudo software-properties-common gnupg git 

RUN apt-get update && apt-get install ninja-build

RUN apt-get update && apt-get install -y cuda-nsight-systems-11-8

RUN apt-get update && apt-get install -y zip 

RUN pip install numpy scipy matplotlib

ENV USER root
ENV HOME /root
# # Switch to user
USER ${USER}

WORKDIR /
RUN wget https://github.com/artempyanykh/marksman/releases/download/2023-12-09/marksman-linux-x64
RUN mv marksman-linux-x64 marksman && chmod +x marksman
RUN mkdir /root/.local
RUN mkdir /root/.local/bin
RUN mv marksman /root/.local/bin
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /

RUN git clone https://github.com/neovim/neovim
WORKDIR /neovim
RUN make CMAKE_BUILD_TYPE=RelWithDebInfo
RUN make install
WORKDIR /
RUN rm -rf /neovim

RUN mkdir /usr/local/nvm
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 21.6.1

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default \
    && npm install -g pyright

RUN pip install autopep8
# This step builds and installs Clang+LLVM toolchain from source that matches the version the 
# Rust compiler uses. Openmp is also enabled.
# LLVM gets installed in opt/llvm
# RUN mkdir /opt/llvm 
# RUN git clone https://github.com/llvm/llvm-project.git
# WORKDIR /llvm-project 
# RUN git checkout llvmorg-16.0.5 
# RUN cmake -S llvm -B build -G Ninja \
#     -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="openmp" -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/opt/llvm
# RUN ninja -C build install

# RUN wget https://www.agner.org/optimize/asmlib.zip
# RUN mkdir /opt/asmlib
# RUN unzip asmlib.zip -d /opt/asmlib
# RUN rm /asmlib.zip

# WORKDIR /
# RUN rm -rf /llvm-project
# ENV LLVM_DIR=/opt/llvm
# ENV LLVM_CONFIG=${LLVM_DIR}/bin/llvm-config
# ENV PATH=${PATH}:${LLVM_DIR}/bin

# # Download FFTW for reverts for the GCC compiler
# RUN wget https://www.fftw.org/fftw-3.3.10.tar.gz
# RUN tar -zxvf fftw-3.3.10.tar.gz
# RUN rm fftw-3.3.10.tar.gz
# WORKDIR /fftw-3.3.10

# # FFTW install for float precision with openmp support
# RUN ./configure --enable-float --enable-openmp 
# # RUN make CC=/opt/llvm/bin/clang -j8 
# RUN make -j8
# RUN make install 

# # FFTW install for double precision with openmp support
# RUN make clean 
# RUN ./configure --enable-openmp
# # RUN make CC=/opt/llvm/bin/clang -j8 
# RUN make -j8
# RUN make install 

# # FFTW install for long double precision with openmp support
# RUN make clean 
# RUN ./configure --enable-long-double --enable-openmp
# # RUN make CC=/opt/llvm/bin/clang -j8 
# RUN make -j8
# RUN make install

# WORKDIR /
# RUN rm -rf /fftw-3.3.10
# # Set user specific environment variables

# jupyter notebook port
# EXPOSE 8888



