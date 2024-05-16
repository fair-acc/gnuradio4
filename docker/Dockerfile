FROM ubuntu:24.04
LABEL authors="akrimm"

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
    EMSDK_VERSION=3.1.44

# Install compilers, package managers and build-tools
# As we are using the the ubuntu prerelease llvm and cmake are fresh enough, readd these if the version is not sufficient anymore
# && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && add-apt-repository 'deb http://apt.llvm.org/mantic/ llvm-toolchain-mantic-18 main'
# && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee -a /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update -y \
    && apt-get install --no-install-recommends -qqy wget gpg ca-certificates software-properties-common bash locales python3-pip npm sudo cmake git make ninja-build clang-18 libc++-18-dev libc++abi-18-dev gdb lldb-18 gcc-14 g++-14 libclang-rt-18-dev python3-dev python3-numpy \
    && locale-gen en_US.UTF-8 && echo 'LANG="en_US.UTF-8"'>/etc/default/locale \
    && useradd -m -g users user \
    && echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user \
    && dpkg-reconfigure --frontend=noninteractive locales && update-locale LANG=en_US.UTF-8 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-18 110 \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && chown -R root:users $EMSDK_HOME && chmod -R 777 $EMSDK_HOME

USER user
ENTRYPOINT ["/bin/bash", "-l", "-c"]
