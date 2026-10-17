FROM ubuntu:26.04
LABEL authors="akrimm"

# test locally via: docker build -f Dockerfile -t test-image:ubuntu26 .

ARG SONAR_SCANNER_HOME=/opt/sonar-scanner
ARG SONAR_SCANNER_VERSION=5.0.1.3006
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
    EMSDK_VERSION=6.0.12

# Compilers, package managers and build tools, all from Ubuntu 26.04 LTS: Clang 21 for the Clang lanes and
# AdaptiveCpp (its newest LLVM, see below), Clang 22 for clang-format, libclang and clang-tidy (devtools). libc++,
# libc++abi and libomp come from Clang 21 only: Ubuntu's libc++-x.y-dev, libc++abi-x.y-dev and libomp-x.y-dev each
# exclude the other versions
RUN apt-get update -y \
    && apt-get install --no-install-recommends -qqy wget gpg ca-certificates software-properties-common bash file locales python3-pip python3-venv npm sudo cmake git make ninja-build \
       clang-21 libc++-21-dev libc++abi-21-dev libclang-21-dev libclang-rt-21-dev libomp-21-dev llvm-21-dev lld-21 \
       clang-22 clang-tools-22 clang-tidy-22 clang-format-22 libclang-22-dev libclang-rt-22-dev llvm-22-dev lld-22 lldb-22 \
       gdb valgrind gcc-15 g++-15 gcc-16 g++-16 ccache mold libcurl4-openssl-dev libngtcp2-dev libnghttp2-dev libnghttp3-dev libcpr-dev gh \
       libonnxruntime-dev libonnxruntime1.23 libonnxruntime-providers onnxruntime-tools libegl1-mesa-dev libgl-dev libgl1-mesa-dri pkg-config \
    && locale-gen en_US.UTF-8 && echo 'LANG="en_US.UTF-8"'>/etc/default/locale \
    && useradd -m -g users user \
    && echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user \
    && dpkg-reconfigure --frontend=noninteractive locales && update-locale LANG=en_US.UTF-8 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-21 110 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 110 --slave /usr/bin/g++ g++ /usr/bin/g++-15 --slave /usr/bin/gcov gcov /usr/bin/gcov-15 \
    && python3 -m pip install --no-cache-dir --no-warn-script-location --break-system-packages gcovr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install sonarcloud-scanner; nodejs is Ubuntu's (22 LTS)
RUN apt-get update -y \
    && apt-get --no-install-recommends -qqy install git unzip curl wget bash fonts-dejavu python3 python3-pip shellcheck nodejs openjdk-17-jre-headless \
    && curl --proto '=https' -fsSL -o /opt/sonar-scanner-cli.zip https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}-linux.zip \
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

# Install Adaptive-cpp: v25.10.0 supports LLVM <= 20, and neither it nor develop builds against LLVM 22 even with
# ACPP_EXPERIMENTAL_LLVM (2026-10-15), so a develop commit tested on LLVM 21, its newest supported LLVM
ARG ACPP_INSTALL_PREFIX=/opt/adaptivecpp
ARG ACPP_GIT_REF=9ec394efa584ced004d644af2194d2133e111d26
ARG ACPP_LLVM_VERSION=21

RUN git init -q /tmp/AdaptiveCpp \
 && git -C /tmp/AdaptiveCpp fetch -q --depth 1 https://github.com/AdaptiveCpp/AdaptiveCpp.git ${ACPP_GIT_REF} \
 && git -C /tmp/AdaptiveCpp checkout -q FETCH_HEAD \
 && cmake -S /tmp/AdaptiveCpp -B /tmp/AdaptiveCpp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_PREFIX} \
      -DACPP_COMPILER_FEATURE_PROFILE=full \
      -DCMAKE_C_COMPILER=/usr/bin/clang-${ACPP_LLVM_VERSION} \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++-${ACPP_LLVM_VERSION} \
      -DLLVM_DIR=/usr/lib/llvm-${ACPP_LLVM_VERSION}/lib/cmake/llvm \
      -DClang_DIR=/usr/lib/llvm-${ACPP_LLVM_VERSION}/lib/cmake/clang \
 && cmake --build /tmp/AdaptiveCpp/build --parallel $(nproc | awk '{print ($1>4)?4:$1}') \
 && cmake --install /tmp/AdaptiveCpp/build \
 && rm -rf /tmp/AdaptiveCpp

ENV PATH="${ACPP_INSTALL_PREFIX}/bin:${PATH}"

# The browser archives installed below are available for amd64 only.

# Browser runtime dependencies.
RUN apt-get update \
    && apt-get install --no-install-recommends -qqy libasound2t64 libgbm1 libgtk-3-0t64 libnss3 libvulkan1 libx11-xcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Refresh the checksums from https://archive.mozilla.org/pub/firefox/releases/<version>/SHA256SUMS
ARG FIREFOX_VERSION=140.12.0esr
ARG FIREFOX_SHA256=3323ee13ac6fe4877fa2e1f4a3aa6b8009f65a620c7bbca96fe86f1a6f433d92

RUN set -eu; \
    if [ "$(dpkg --print-architecture)" != amd64 ]; then echo 'skipping firefox: amd64-only image feature'; exit 0; fi; \
    curl --proto '=https' -fsSLo /tmp/firefox.tar.xz "https://archive.mozilla.org/pub/firefox/releases/${FIREFOX_VERSION}/linux-x86_64/en-US/firefox-${FIREFOX_VERSION}.tar.xz"; \
    echo "${FIREFOX_SHA256}  /tmp/firefox.tar.xz" | sha256sum --check -; \
    mkdir -p "/opt/firefox-${FIREFOX_VERSION}/distribution"; \
    tar -xJf /tmp/firefox.tar.xz --strip-components=1 -C "/opt/firefox-${FIREFOX_VERSION}"; \
    printf '{"policies":{"DisableAppUpdate":true,"DisableTelemetry":true}}\n' > "/opt/firefox-${FIREFOX_VERSION}/distribution/policies.json"; \
    ln -s "/opt/firefox-${FIREFOX_VERSION}/firefox" /usr/local/bin/firefox; \
    rm -f /tmp/firefox.tar.xz; \
    firefox --version

ARG CHROME_VERSION=152.0.7977.42
ARG CHROME_SHA256=cb77f4781cad7d5e06fcc78b4476e6a6375616e7278dc313abaa9db22ed4674e

RUN set -eu; \
    if [ "$(dpkg --print-architecture)" != amd64 ]; then echo 'skipping chrome: amd64-only image feature'; exit 0; fi; \
    curl --proto '=https' -fsSLo /tmp/chrome.zip "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chrome-linux64.zip"; \
    echo "${CHROME_SHA256}  /tmp/chrome.zip" | sha256sum --check -; \
    unzip -q /tmp/chrome.zip -d /opt; \
    mv /opt/chrome-linux64 "/opt/chrome-${CHROME_VERSION}"; \
    ln -s "/opt/chrome-${CHROME_VERSION}/chrome" /usr/local/bin/google-chrome; \
    ln -s /usr/local/bin/google-chrome /usr/local/bin/chrome; \
    rm -f /tmp/chrome.zip; \
    google-chrome --version

ENV FIREFOX_BIN=/usr/local/bin/firefox \
    CHROME_BIN=/usr/local/bin/google-chrome
# Pre-build ONNX Runtime static libraries for WASM and native fallback.
# The system .so (libonnxruntime1.23) is used for native builds by default;
# these static libs provide WASM support and a native fallback without .so.
ARG ONNXRUNTIME_VERSION=1.23.2
ARG ORT_STATIC_LIB_DIR=/opt/onnxruntime

COPY build_onnxruntime.sh /usr/local/bin/build_onnxruntime.sh
RUN chmod +x /usr/local/bin/build_onnxruntime.sh \
 && ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION} ORT_STATIC_LIB_DIR=${ORT_STATIC_LIB_DIR} \
    EMSDK_HOME=${EMSDK_HOME} /usr/local/bin/build_onnxruntime.sh

ENV ORT_STATIC_LIB_DIR=${ORT_STATIC_LIB_DIR}

USER user
ENTRYPOINT ["/bin/bash", "-l", "-c"]
