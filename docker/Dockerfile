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
    && apt-get install --no-install-recommends -qqy wget gpg ca-certificates software-properties-common bash file locales python3-pip npm sudo cmake git make ninja-build clang-20 clang-tools-20 libc++-20-dev libc++abi-20-dev libclang-20-dev libclang-rt-20-dev libomp-20-dev llvm-20-dev lld-20 gdb lldb-20 gcc-14 g++-14 gcc-15 g++-15 ccache mold libcurl4-openssl-dev libngtcp2-dev libnghttp2-dev libnghttp3-dev libonnxruntime-dev libonnxruntime1.21 libonnxruntime-providers onnxruntime-tools libegl1-mesa-dev libgl-dev \
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
 && cmake --build /tmp/AdaptiveCpp/build --parallel $(nproc | awk '{print ($1>4)?4:$1}') \
 && cmake --install /tmp/AdaptiveCpp/build \
 && rm -rf /tmp/AdaptiveCpp

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y libcpr-dev

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
# The system .so (libonnxruntime1.21) is used for native builds by default;
# these static libs provide WASM support and a native fallback without .so.
ARG ONNXRUNTIME_VERSION=1.21.0
ARG ORT_STATIC_LIB_DIR=/opt/onnxruntime

COPY build_onnxruntime.sh /usr/local/bin/build_onnxruntime.sh
RUN chmod +x /usr/local/bin/build_onnxruntime.sh \
 && ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION} ORT_STATIC_LIB_DIR=${ORT_STATIC_LIB_DIR} \
    EMSDK_HOME=${EMSDK_HOME} /usr/local/bin/build_onnxruntime.sh

ENV ORT_STATIC_LIB_DIR=${ORT_STATIC_LIB_DIR}

USER user
ENTRYPOINT ["/bin/bash", "-l", "-c"]
