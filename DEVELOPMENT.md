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
- optional for compiling to webassembly: emscripten >= 5.0.0

To apply the project's formatting rules, you'll also need the correct formatters, `clang-format-18` and `cmake-format`. With these installed you can use the scripts in the repository to reformat your changes. For smaller changes, the CI will provide you with a patch which will fix the formatting (click on the "Details" link on the failed Restyled.io check), but for bigger changes it's useful to have local formatting.

Once these are installed, you should be able to just compile and run GNURadio4:

```bash
me@host$ cd gnuradio4
me@host$ cmake -S . -B build
me@host$ cmake --build build
me@host$ ctest --test-dir build
```

### Win32 Development Environment - MSYS2

The current development environment in Windows uses `MSYS2`, specifically `UCRT64` and `CLANG64` to allow for building gnuradio4.
While this is not the desired end development environment for Windows, it currently builds and runs the testsuite.
To set up the `MSYS2` environment, navigate to https://www.msys2.org and download the installer, currently `msys2-x86_64-20251213.exe`.
`MSYS2` is a rolling release, so I hope these instructions continue to work as the code gets updated.
Chances are new build packages will cause small breakages but hopefully not insurmountable ones.

To install `msys2`, follow the instructions on https://www.msys2.org.
Once installed, open either the `UCRT64` or `CLANG64` environment and update the environment via the pacman package installer.

```bash
me@host UCRT64
$ pacman -Syu
```

This will likely require the closing of the terminal and reopening it.
Once the terminal is reopened, we should install the development programs.
The minimal requirement would be the development programs for `UCRT64` as follows.

```bash
me@host UCRT64
$ pacman -S git moreutils \
                    mingw-w64-ucrt-x86_64-ccache \
                    mingw-w64-ucrt-x86_64-toolchain \
                    mingw-w64-ucrt-x86_64-python-numpy \
                    mingw-w64-ucrt-x86_64-cmake \
                    mingw-w64-ucrt-x86_64-ninja \
                    mingw-w64-ucrt-x86_64-clang-tools-extra \
                    mingw-w64-ucrt-x86_64-dlfcn \
                    mingw-w64-ucrt-x86_64-nodejs \
                    mingw-w64-ucrt-x86_64-soapysdr \
                    mingw-w64-ucrt-x86_64-soapyrtlsdr
```

If one wants to use `CLANG64` to build instead of `UCRT64`, you can use the comamnd that follows.

```bash
me@host CLANG64
$ pacman -S git moreutils \
                    mingw-w64-clang-x86_64-ccache \
                    mingw-w64-clang-x86_64-toolchain \
                    mingw-w64-clang-x86_64-python-numpy \
                    mingw-w64-clang-x86_64-cmake \
                    mingw-w64-clang-x86_64-ninja \
                    mingw-w64-clang-x86_64-clang-tools-extra \
                    mingw-w64-clang-x86_64-dlfcn \
                    mingw-w64-clang-x86_64-nodejs \
                    mingw-w64-clang-x86_64-soapysdr \
                    mingw-w64-clang-x86_64-soapyrtlsdr
```

And of course, both can be run to install both environments.
Of note the `MINGW64` environment is not being used.
The only difference between it and `UCRT64` is the use of the `msvcrt` C library instead of `ucrt`.
Nonetheless, there were problems building for this environment, so instructions for it are not included in this document.

Initial build for ucrt64 and clang64 environments.
We will need to set `-DWARNINGS_AS_ERRORS=OFF` as both `g++` and `clang++` generate warnings during the build.
Also, we need to set `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to be able to use the full features of the language server protocol of `nvim`.

```bash
me@host CLANG64
$ cmake -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_VERBOSE_MAKEFILE=ON \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DWARNINGS_AS_ERRORS=OFF \
            -DCMAKE_INSTALL_PREFIX=/home/$USER/gr4 \
            -S . -B build
```

Then follow the rest of the build steps outlined above.
Build gnuradio4 by running the following:

```bash
me@host CLANG64
$ cmake --build build
```

Once the system is built, run the testsuite:

```bash
me@host CLANG64
$ ctest --timeout 120 --test-dir build
```

#### Neovim based Development Environment

The following optional section describes a way to set up a development environment with code completion and formatting support based on neovim.

```bash
me@host CLANG64
$ pacman -S mingw-w64-clang-x86_64-neovim
```

To configure `nvim` to properly use `clangd` and format your work properly, some modifications must be made to the end of your `.bashrc`.
Add the following code snippet:

```bash
export LANG=en_US.UTF-8

# Set history
shopt -s histappend
export HISTSIZE=999999
export HISTFILESIZE=999999
HISTCONTROL=erasedups
PROMPT_COMMAND="history -w; $PROMPT_COMMAND"
tac $HISTFILE | awk '!x[$0]++' | tac | sponge $HISTFILE

export EDITOR=nvim

export XDG_DATA_HOME="$HOME/.local/share"
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_STATE_HOME="$HOME/.local/state"
export XDG_RUNTIME_DIR="/tmp/$USER-runtime-dir"
```

Now close the terminal and reopen a new one to update the bash environment variables.
The next program to configure is `neovim` or `nvim`.
To configure it, make the directory `$HOME/.config/nvim` and put the following script into it as `init.lua`.
When `nvim` is launched after the script is put in place, it will download and setup `lazy.nvim`, `mason.nvim`, `mason-lspconfig`, `nvim-lspconfig`, and `nvim-cmp`.
These helper programs or scripts for `nvim` allow it to use `clangd`, perform autocompletion, and load headers or examine functions by making a pair of keystrokes (gd) over the header or function name in the file you're editing.
It also allows for formatting CMakeLists.txt files properly.

```bash
-- Bootstrap lazy.nvim
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git", "clone", "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)

-- Install and configure plugins
require("lazy").setup({
  -- Mason, keeping for other tools, but not clangd
  {
    "williamboman/mason.nvim",
    config = true,
  },

  -- Mason-lspconfig setup
  {
    "williamboman/mason-lspconfig.nvim",
    dependencies = { "williamboman/mason.nvim" },
    config = function()
      require("mason-lspconfig").setup({
        ensure_installed = {},
      })
    end,
  },

  -- LSP setup
  {
    "neovim/nvim-lspconfig",
  },

  -- Autocompletion
  {
    "hrsh7th/nvim-cmp",
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
      "hrsh7th/cmp-buffer",
      "hrsh7th/cmp-path",
      "hrsh7th/cmp-cmdline",
      "L3MON4D3/LuaSnip",
      "saadparwaiz1/cmp_luasnip",
    },
    config = function()
      local cmp = require("cmp")
      local luasnip = require("luasnip")

      cmp.setup({
        snippet = {
          expand = function(args)
            luasnip.lsp_expand(args.body)
          end,
        },
        mapping = cmp.mapping.preset.insert({
          ["<Tab>"]   = cmp.mapping.select_next_item(),
          ["<S-Tab>"] = cmp.mapping.select_prev_item(),
          ["<CR>"]    = cmp.mapping.confirm({ select = true }),
        }),
        sources = {
          { name = "nvim_lsp" },
          { name = "luasnip" },
          { name = "buffer" },
          { name = "path" },
        },
      })
    end,
  },
})

-- lsp keybindings
local on_attach = function(_, bufnr)
  local opts = { noremap = true, silent = true, buffer = bufnr }

  vim.keymap.set("n", "gd", vim.lsp.buf.definition, opts)
  vim.keymap.set("n", "gD", vim.lsp.buf.declaration, opts)
  vim.keymap.set("n", "gi", vim.lsp.buf.implementation, opts)
  vim.keymap.set("n", "gr", vim.lsp.buf.references, opts)
  vim.keymap.set("n", "K",  vim.lsp.buf.hover, opts)
  vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename, opts)
  vim.keymap.set("n", "<leader>ca", vim.lsp.buf.code_action, opts)
  vim.keymap.set("n", "[d", vim.diagnostic.goto_prev, opts)
  vim.keymap.set("n", "]d", vim.diagnostic.goto_next, opts)

  vim.api.nvim_create_autocmd("BufWritePre", {
    buffer = bufnr,
    callback = function()
      vim.lsp.buf.format({ async = false })
    end,
  })
end

-- Setup clangd
vim.api.nvim_create_autocmd("FileType", {
  pattern = { "c", "cpp", "objc", "objcpp" },
  callback = function()
    vim.lsp.start({
      name = "clangd",
      cmd = {
        "clangd",
        "--clang-tidy",
        "--header-insertion=never",
        "--fallback-style=LLVM",
        "--background-index",
        "--completion-style=detailed",
        "--cross-file-rename",
        "--suggest-missing-includes",
      },

      root_dir = vim.fs.root(0, {
        "compile_commands.json",
        "compile_flags.txt",
        ".clangd",
        ".git",
      }) or vim.fn.getcwd(),

      capabilities = require("cmp_nvim_lsp").default_capabilities(),
      on_attach = on_attach,

      init_options = {
        fallbackFlags = { "-std=c++17" },
        clangdFileStatus = true,
      },
    })
  end,
})

-- Function to go to the last cursor position
vim.api.nvim_create_autocmd("BufReadPost", {
  pattern = "*",
  callback = function()
    local last_pos = vim.fn.line("'\"")
    if last_pos > 0 and last_pos <= vim.fn.line("$") then
      vim.cmd('normal! g`"')
    end
  end,
})

-- Autocommand to trigger the fuction when a buffer is read
vim.api.nvim_create_autocmd("BufWritePost", {
  pattern = { "*.cmake", "CMakeLists.txt" },
  callback = function()
    -- Get full path with proper escaping
    local file = vim.fn.expand("%:p")

    -- Use raw vim.fm.system to call cmake-format directly
    local result = vim.fn.system({ "cmake-format", "-i", file })

    -- Show stderr or error
    if vim.v.shell_error ~= 0 then
      vim.notify("cmake-format failed:\n" .. result, vim.log.levels.ERROR)
    else
      vim.cmd("edit!")
    end
  end,
})
```

Once the `compile_commands.json` is created in the cmake configure step, copy it to the root gnuradio directory so it can be used by nvim.

```bash
me@host CLANG64
$ cp build/compile_commands.json .
```

To improve the fonts in the `mintty` edit `$HOME/.minttyrc` and add the following to it. This step is cosmetic and not strictly necessary.

```bash
FontHeight=9
Font=DejaVu Sans Mono
Locale=en_US
Charset=UTF-8
CtrlShiftShortcuts=no
CursorType=block
```
