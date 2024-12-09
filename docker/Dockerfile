FROM ubuntu:25.10
LABEL authors="akrimm"

# test locally via: docker build -f Dockerfile -t test-image:ubuntu25 .

ARG SONAR_SCANNER_HOME=/opt/sonar-scanner
ARG SONAR_SCANNER_VERSION=5.0.1.3006
ARG NODE_MAJOR=18
ENV JAVA_HOME=/opt/java/openjdk \
    HOME=/tmp \
    XDG_CONFIG_HOME=/tmp \
    SONAR_SCANNER_HOME=${SONAR_SCANNER_HOME} \
    SONAR_USER_HOME=${SONAR_SCANNER_HOME}/.sonar \
    PATH=${SONAR_SCANNER_HOME}/bin:${PATH} \
    NODE_PATH=/usr/local/lib/node_modules \
    SRC_PATH=/usr/src \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    CMAKE_MAKE_PROGRAM=ninja \
    EMSDK_HOME=/opt/emsdk \
    EMSDK_VERSION=5.0.2

# Install compilers, package managers and build-tools
# As we are using the the ubuntu prerelease llvm and cmake are fresh enough, readd these if the version is not sufficient anymore
# && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && add-apt-repository 'deb http://apt.llvm.org/mantic/ llvm-toolchain-mantic-18 main'
# && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee -a /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update -y \
    && apt-get install --no-install-recommends -qqy wget gpg ca-certificates software-properties-common bash file locales python3-pip npm sudo cmake git make ninja-build clang-20 clang-tools-20 libc++-20-dev libc++abi-20-dev libclang-20-dev libclang-rt-20-dev libomp-20-dev llvm-20-dev lld-20 gdb lldb-20 gcc-14 g++-14 gcc-15 g++-15 ccache mold libcurl4-openssl-dev libngtcp2-dev libnghttp2-dev libnghttp3-dev libonnxruntime-dev libonnxruntime1.21 libonnxruntime-providers onnxruntime-tools \
    && locale-gen en_US.UTF-8 && echo 'LANG="en_US.UTF-8"'>/etc/default/locale \
    && useradd -m -g users user \
    && echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user \
    && dpkg-reconfigure --frontend=noninteractive locales && update-locale LANG=en_US.UTF-8 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-20 110 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 110 --slave /usr/bin/g++ g++ /usr/bin/g++-14 --slave /usr/bin/gcov gcov /usr/bin/gcov-14 \
    && python3 -m pip install --no-cache-dir --no-warn-script-location --break-system-packages gcovr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install sonarcloud-scanner
# nodejs should be fresh enough
#apt-get -qqy update && apt-get --no-install-recommends -qqy install ca-certificates curl gnupg  && \
#    mkdir -p /etc/apt/keyrings && \
#    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
#    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
#    apt-get -qqy update && \
RUN apt-get update -y \
    && apt-get --no-install-recommends -qqy install git unzip curl wget bash fonts-dejavu python3 python3-pip shellcheck nodejs openjdk-17-jre-headless \
    && curl -fsSL -o /opt/sonar-scanner-cli.zip https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}-linux.zip \
    && unzip /opt/sonar-scanner-cli.zip \
    && rm /opt/sonar-scanner-cli.zip \
    && mv sonar-scanner-${SONAR_SCANNER_VERSION}-linux ${SONAR_SCANNER_HOME} \
    && pip install --no-cache-dir --break-system-packages pylint \
    && mkdir -p "${SRC_PATH}" "${SONAR_USER_HOME}" "${SONAR_USER_HOME}/cache" \
    && chmod -R 555 "${SONAR_SCANNER_HOME}" && chmod -R 777 $SONAR_USER_HOME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install emscripten sdk
RUN apt-get update -y \
    && apt-get install --no-install-recommends -qqy bzip2 xz-utils \
    && mkdir -p $EMSDK_HOME \
    && git clone --depth=1 https://github.com/emscripten-core/emsdk.git $EMSDK_HOME \
    && cd $EMSDK_HOME \
    && ./emsdk install $EMSDK_VERSION \
    && ./emsdk activate $EMSDK_VERSION \
    && npm install --ignore-scripts -g xhr2 \
    && ./upstream/emscripten/embuilder build sdl3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && chown -R root:users $EMSDK_HOME && chmod -R 777 $EMSDK_HOME

# Install GitHub command-line interface (gh)
# Instructions copied from https://github.com/cli/cli/blob/trunk/docs/install_linux.md
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && curl -s https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Adaptive-cpp
ARG ACPP_INSTALL_PREFIX=/opt/adaptivecpp
ARG ACPP_GIT_REF=v25.10.0

RUN git clone --depth 1 --branch ${ACPP_GIT_REF} https://github.com/AdaptiveCpp/AdaptiveCpp.git /tmp/AdaptiveCpp \
 && cmake -S /tmp/AdaptiveCpp -B /tmp/AdaptiveCpp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_PREFIX} \
      -DACPP_COMPILER_FEATURE_PROFILE=full \
      -DCMAKE_C_COMPILER=/usr/bin/clang-20 \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 \
      -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
      -DClang_DIR=/usr/lib/llvm-20/lib/cmake/clang \
 && cmake --build /tmp/AdaptiveCpp/build \
 && cmake --install /tmp/AdaptiveCpp/build \
 && rm -rf /tmp/AdaptiveCpp

ENV PATH="${ACPP_INSTALL_PREFIX}/bin:${PATH}"

# Pre-build ONNX Runtime static libraries for WASM and native fallback.
# The system .so (libonnxruntime1.21) is used for native builds by default;
# these static libs provide WASM support and a native fallback without .so.
ARG ONNXRUNTIME_VERSION=1.21.0
ARG ORT_STATIC_LIB_DIR=/opt/onnxruntime

RUN git clone --depth 1 --branch v${ONNXRUNTIME_VERSION} \
      https://github.com/microsoft/onnxruntime.git /tmp/ort-src \
 && mkdir -p ${ORT_STATIC_LIB_DIR}/wasm/lib ${ORT_STATIC_LIB_DIR}/wasm/include \
             ${ORT_STATIC_LIB_DIR}/native/lib ${ORT_STATIC_LIB_DIR}/native/include \
 && cd /tmp/ort-src \
 # --- GCC 15 patches ---
 && SEMVER_H=onnxruntime/core/common/semver.h \
 && if [ -f "$SEMVER_H" ] && ! grep -q '<cstdint>' "$SEMVER_H"; then \
      sed -i '/#pragma once/a #include <cstdint>' "$SEMVER_H"; \
      sed -i 's/uint32_t major{}/uint32_t ver_major{}/g' "$SEMVER_H"; \
      sed -i 's/uint32_t minor{}/uint32_t ver_minor{}/g' "$SEMVER_H"; \
      sed -i 's/uint32_t patch{}/uint32_t ver_patch{}/g' "$SEMVER_H"; \
    fi \
 && SEMVER_CC=onnxruntime/core/common/semver.cc \
 && if [ -f "$SEMVER_CC" ] && grep -qE '\.(major|minor|patch)' "$SEMVER_CC"; then \
      sed -i 's/->major/->ver_major/g; s/->minor/->ver_minor/g; s/->patch/->ver_patch/g' "$SEMVER_CC"; \
      sed -i 's/\.major/\.ver_major/g; s/\.minor/\.ver_minor/g; s/\.patch/\.ver_patch/g' "$SEMVER_CC"; \
    fi \
 # --- Eigen hash fixes ---
 && if [ -f cmake/deps.txt ]; then \
      sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g' cmake/deps.txt; \
      sed -i 's/5ea4d05e62d7f954a46b3213f9b2535bdd866803/51982be81bbe52572b54180454df11a3ece9a934/g' cmake/deps.txt; \
    fi \
 # --- build native static lib ---
 && python3 tools/ci_build/build.py \
      --build_dir /tmp/ort-build-native --config Release \
      --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
      --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_Protobuf=TRUE \
      --cmake_extra_defines CMAKE_DISABLE_FIND_PACKAGE_flatbuffers=TRUE \
      --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
      --update --build --minimal_build extended \
      --disable_ml_ops --disable_contrib_ops \
      --compile_no_warning_as_error --skip_tests --parallel 6 \
 && B=/tmp/ort-build-native/Release \
 && MRI=$(mktemp) \
 && echo "CREATE ${ORT_STATIC_LIB_DIR}/native/lib/libonnxruntime.a" > "$MRI" \
 && for lib in libonnxruntime_common.a libonnxruntime_flatbuffers.a libonnxruntime_framework.a \
               libonnxruntime_graph.a libonnxruntime_lora.a libonnxruntime_mlas.a \
               libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_session.a \
               libonnxruntime_util.a; do \
      [ -f "$B/$lib" ] && echo "ADDLIB $B/$lib" >> "$MRI"; \
    done \
 && for lib in "$B/_deps/onnx-build/libonnx.a" "$B/_deps/onnx-build/libonnx_proto.a"; do \
      [ -f "$lib" ] && echo "ADDLIB $lib" >> "$MRI"; \
    done \
 && for lib in "$B/_deps/protobuf-build/libprotobuf.a" "$B/_deps/protobuf-build/libprotobuf-lite.a"; do \
      [ -f "$lib" ] && { echo "ADDLIB $lib" >> "$MRI"; break; }; \
    done \
 && [ -f "$B/_deps/flatbuffers-build/libflatbuffers.a" ] && \
      echo "ADDLIB $B/_deps/flatbuffers-build/libflatbuffers.a" >> "$MRI" \
 && find "$B/_deps/abseil_cpp-build" -name "libabsl_*.a" -type f 2>/dev/null | sort | \
      while read -r lib; do echo "ADDLIB $lib" >> "$MRI"; done \
 && [ -f "$B/_deps/pytorch_cpuinfo-build/libcpuinfo.a" ] && \
      echo "ADDLIB $B/_deps/pytorch_cpuinfo-build/libcpuinfo.a" >> "$MRI" \
 && for re2 in "$B/_deps/re2-build/libre2.a" /usr/local/lib64/libre2.a /usr/lib/x86_64-linux-gnu/libre2.a; do \
      [ -f "$re2" ] && { echo "ADDLIB $re2" >> "$MRI"; break; }; \
    done \
 && echo "SAVE" >> "$MRI" && echo "END" >> "$MRI" \
 && ar -M < "$MRI" && rm -f "$MRI" \
 && strip --strip-debug ${ORT_STATIC_LIB_DIR}/native/lib/libonnxruntime.a 2>/dev/null || true \
 && cp -r /tmp/ort-src/include/onnxruntime/core/session/* ${ORT_STATIC_LIB_DIR}/native/include/ \
 # --- build WASM static lib ---
 # ORT's build.py bundles its own emsdk (4.0.8); replace with system emsdk
 # to ensure matching LLVM versions between ORT objects and wasm-ld
 && rm -rf cmake/external/emsdk && ln -s ${EMSDK_HOME} cmake/external/emsdk \
 && . ${EMSDK_HOME}/emsdk_env.sh \
 && python3 tools/ci_build/build.py \
      --build_dir /tmp/ort-build-wasm --config Release \
      --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      --cmake_extra_defines FETCHCONTENT_QUIET=OFF \
      --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
      --cmake_extra_defines onnxruntime_ENABLE_LTO=OFF \
      --cmake_extra_defines "CMAKE_CXX_FLAGS=-fwasm-exceptions" \
      --cmake_extra_defines "CMAKE_C_FLAGS=-fwasm-exceptions" \
      --update --build --minimal_build extended \
      --disable_ml_ops --disable_contrib_ops \
      --compile_no_warning_as_error --skip_tests --parallel 6 \
      --skip_submodule_sync \
      --build_wasm_static_lib --enable_wasm_simd --enable_wasm_threads \
      --disable_wasm_exception_catching --enable_wasm_api_exception_catching \
 && WB=/tmp/ort-build-wasm/Release \
 && if [ -f "$WB/libonnxruntime_webassembly.a" ]; then \
      cp "$WB/libonnxruntime_webassembly.a" ${ORT_STATIC_LIB_DIR}/wasm/lib/; \
    elif [ -f "$WB/onnxruntime_webassembly.a" ]; then \
      cp "$WB/onnxruntime_webassembly.a" ${ORT_STATIC_LIB_DIR}/wasm/lib/libonnxruntime_webassembly.a; \
    fi \
 && cp -r /tmp/ort-src/include/onnxruntime/core/session/* ${ORT_STATIC_LIB_DIR}/wasm/include/ \
 # --- cleanup ---
 && rm -rf /tmp/ort-src /tmp/ort-build-native /tmp/ort-build-wasm

ENV ORT_STATIC_LIB_DIR=${ORT_STATIC_LIB_DIR}

USER user
ENTRYPOINT ["/bin/bash", "-l", "-c"]
