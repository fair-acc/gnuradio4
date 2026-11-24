// Drives headless Chrome over the DevTools protocol (CDP) to run and inspect Emscripten builds in a real browser.
// GR4_BROWSER_GPU=hardware gives Chrome the GPU's own WebGPU adapter beside SwiftShader (Vulkan, headless on Linux:
// AMD RDNA 3, NVIDIA RTX 3070).
// Serving goes through serve.mjs, which sets the COOP/COEP headers that SharedArrayBuffer and pthreads need.
// Waits are for a named condition with a deadline, never a fixed duration.

import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const kBrowserCandidates = [process.env.CHROME_BIN, "google-chrome", "chromium", "chromium-browser", "google-chrome-stable"].filter(Boolean);
// without the last two NVIDIA gets no adapter: the headless compositor fails on the Vulkan surface
const kAngleArguments =
  process.env.GR4_BROWSER_GPU === "hardware"
    ? ["--enable-features=Vulkan,VulkanFromANGLE", "--use-angle=vulkan", "--ignore-gpu-blocklist", "--disable-vulkan-surface"]
    : ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"];
const kDefaultTimeoutMs = 60_000;

class Deadline {
  constructor(milliseconds, what) {
    this.expiry = Date.now() + milliseconds;
    this.what = what;
  }

  check() {
    if (Date.now() > this.expiry) {
      throw new Error(`timed out waiting for ${this.what}`);
    }
  }
}

const untilReachable = async (probe, what, timeoutMs = kDefaultTimeoutMs) => {
  const deadline = new Deadline(timeoutMs, what);
  for (;;) {
    try {
      const value = await probe();
      if (value) {
        return value;
      }
    } catch {
      // not up yet
    }
    deadline.check();
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
};

// for a wait no condition can be polled for (a socket opening, an answer): a page whose main thread is blocked never answers
const withinDeadline = (promise, what, timeoutMs = kDefaultTimeoutMs) => {
  let timer = null;
  const expired = new Promise((_, reject) => (timer = setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), timeoutMs)));
  return Promise.race([promise, expired]).finally(() => clearTimeout(timer));
};

export class NoBrowserError extends Error {}

/// serves `directories`, searched in order for each request
export const startServer = async (directories) => {
  const server = spawn(
    process.execPath,
    [path.join(import.meta.dirname, "serve.mjs"), "--port", "0", ...directories.flatMap((directory) => ["--directory", directory])],
    {
      stdio: ["ignore", "pipe", "inherit"],
    },
  );
  let output = "";
  const reported = new Promise((resolve, reject) => {
    server.stdout.on("data", (chunk) => {
      output += chunk;
      const match = /listening on port (\d+)/.exec(output);
      if (match) {
        resolve(Number(match[1]));
      }
    });
    server.on("exit", (code) => reject(new Error(`serve.mjs exited with ${code} before reporting a port`)));
  });
  try {
    const port = await withinDeadline(reported, "serve.mjs to report its port");
    return { origin: `http://localhost:${port}`, stop: () => server.kill() };
  } catch (error) {
    server.kill(); // it shares ctest's stderr, which ctest reads until every writer has gone
    throw error;
  }
};

let browsersLaunched = 0;
export const launchBrowser = async (browserArguments = []) => {
  const profile = await mkdtemp(path.join(tmpdir(), "gr4-browser-"));
  // one port per test process and browser, so parallel ctest jobs do not collide
  const port = 9300 + ((process.pid + 101 * browsersLaunched++) % 500);
  // Chrome's sandbox cannot start in a container (the CI image runs as an unprivileged user), and only pages served here are opened
  const containerArguments = ["--no-sandbox", "--disable-dev-shm-usage"];
  let browser = null;
  for (const candidate of kBrowserCandidates) {
    browser = spawn(
      candidate,
      [
        "--headless=new",
        "--use-gl=angle",
        ...kAngleArguments,
        "--enable-unsafe-webgpu", // Linux Chrome exposes navigator.gpu only behind this flag
        "--no-first-run",
        "--no-default-browser-check",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1280,800",
        ...containerArguments,
        `--user-data-dir=${profile}`,
        `--remote-debugging-port=${port}`,
        ...browserArguments,
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    const started = await new Promise((resolve) => {
      browser.once("error", () => resolve(false));
      browser.once("spawn", () => resolve(true));
    });
    if (started) {
      break;
    }
    browser = null;
  }
  if (!browser) {
    await rm(profile, { recursive: true, force: true });
    throw new NoBrowserError(`no headless browser found, tried: ${kBrowserCandidates.join(", ")}`);
  }
  let errorOutput = "";
  browser.stderr.on("data", (chunk) => {
    errorOutput = (errorOutput + chunk).slice(-4096);
  });
  const exited = new Promise((resolve) => browser.once("exit", (code, signal) => resolve({ code, signal })));
  const endpoint = `http://127.0.0.1:${port}`;
  try {
    await Promise.race([
      untilReachable(() => fetch(`${endpoint}/json/version`).then((response) => response.ok), "the browser's debugging endpoint"),
      exited.then(({ code, signal }) => {
        throw new Error(`the browser exited (${signal ?? code}) before its debugging endpoint came up`);
      }),
    ]);
  } catch (error) {
    browser.kill("SIGKILL");
    await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    throw new Error(`${error.message}; its last output:\n${errorOutput}`);
  }
  return {
    endpoint,
    stop: async () => {
      browser.kill("SIGKILL");
      await exited;
      // the browser is still flushing its cache as it dies, so a single rmdir races it
      await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    },
  };
};

export const openPage = async (browser, url) => {
  const opened = fetch(`${browser.endpoint}/json/new?${encodeURIComponent(url)}`, { method: "PUT" }).then((response) => response.json());
  const target = await withinDeadline(opened, "the browser to open a page");
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  const pending = new Map();
  const consoleLines = [];
  let nextId = 0;

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
    if (message.method === "Runtime.consoleAPICalled") {
      consoleLines.push(message.params.args.map((argument) => argument.value ?? argument.description ?? "").join(" "));
    }
    if (message.method === "Runtime.exceptionThrown") {
      const details = message.params.exceptionDetails;
      consoleLines.push(`uncaught: ${details.exception?.description ?? details.text}`);
    }
  };
  await withinDeadline(new Promise((resolve) => (socket.onopen = resolve)), "the page's DevTools socket to open");

  const command = (method, params = {}) =>
    withinDeadline(
      new Promise((resolve) => {
        const id = ++nextId;
        pending.set(id, resolve);
        socket.send(JSON.stringify({ id, method, params }));
      }),
      `the browser's answer to ${method}`,
    );
  await command("Runtime.enable");

  const evaluate = async (expression) => {
    const message = await command("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    const thrown = message.result?.exceptionDetails;
    if (thrown) {
      throw new Error(`evaluating ${expression}: ${thrown.exception?.description ?? thrown.text}`);
    }
    return message.result?.result?.value;
  };

  return {
    evaluate,
    consoleLines,
    waitFor: (expression, timeoutMs = kDefaultTimeoutMs) => untilReachable(() => evaluate(expression), expression, timeoutMs),
    screenshot: async (file) => {
      const shot = await command("Page.captureScreenshot", { format: "png" });
      await writeFile(file, Buffer.from(shot.result.data, "base64"));
    },
  };
};

/// runs `body` against `directories` served and a fresh headless browser, tearing both down whatever happens, a SIGTERM included
export const withBrowser = async (directories, body, { browserArguments = [] } = {}) => {
  const server = await startServer(directories);
  let browser = null;
  const stopOnSignal = async () => {
    await browser?.stop();
    server.stop();
    process.exit(143);
  };
  process.once("SIGTERM", stopOnSignal);
  try {
    browser = await launchBrowser(browserArguments);
    return await body({ server, browser });
  } finally {
    process.off("SIGTERM", stopOnSignal);
    await browser?.stop();
    server.stop();
  }
};
