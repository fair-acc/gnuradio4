# GNURadio 4.0 Development Environment

## Getting the source

Get the source from the GitHub Repository:
```bash
git clone git@github.com:fair-acc/gnuradio4.git
```

## Building
### Docker CLI
To just compile GNURadio4 without installing any dependencies you can just use the Docker image which is also used by our CI builds. The snippet below uses `docker run` to start the container with the current directory mapped into the container with the correct user and group IDs.
It then compiles the project and runs the testsuite.
Note that while the binaries inside of `./build` can be accessed on the host system, they are linked against the libraries of the container and will most probably not run on the host system.

```bash
me@host$ cd gnuradio4
me@host$ docker run \
    --user `id -u`:`id -g` \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --workdir=/work/src --volume `pwd`:/work/src -it \
    ghcr.io/fair-acc/gr4-build-container bash

me@aba123ef$ # export CXX=c++ # uncomment to use clang
me@aba123ef$ cmake -S . -B build
me@aba123ef$ cmake --build build
me@aba123ef$ ctest --test-dir build
```
### Docker IDE
Some IDEs provide a simple way to specify a docker container to use for building and executing a project. For example in JetBrains CLion you can set this up in `Settings->Build,Execution,Deployment->Toolchains->[+]->Docker`, leaving everything as the default except for setting `Image` to `ghcr.io/fair-acc/gr4-build-container`.
By default this will use the gcc-14 compiler included in the image, by setting `CXX` to `clang++-18` you can also use clang.

### Native
To be able to natively compile some prerequisites have to be installed:
- gcc >= 13 and/or clang >= 17
- cmake >= 3.25.0
- ninja (or GNU make)
- optional for python block support: python3
- optional for soapy (limesdr,rtlsdr) blocks: soapysdr
- optional for compiling to webassembly: emscripten >= 3.1.50

To apply the project's formatting rules, you'll also need the correct formatters, `clang-format-18` and `cmake-format`. With these installed you can use the scripts in the repository to reformat your changes. For smaller changes, the CI will provide you with a patch which will fix the formatting (click on the "Details" link on the failed Restyled.io check), but for bigger changes it's useful to have local formatting.

Once these are installed, you should be able to just compile and run GNURadio4:

```bash
me@host$ cd gnuradio4
me@host$ cmake -S . -B build
me@host$ cmake --build build
me@host$ ctest --test-dir build

