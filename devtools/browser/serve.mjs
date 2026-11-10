#!/usr/bin/env node
// Serves Emscripten builds with the COOP/COEP headers that SharedArrayBuffer, and so pthreads, need.
// SharedArrayBuffer also needs a secure context, and only localhost is exempt.

import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import path from "node:path";

const kDefaultPort = 8000;

const kContentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".data": "application/octet-stream",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".wav": "audio/wav",
};

const usage = `usage: serve.mjs [options]

  -d, --directory <dir>  directory to serve; repeated, each request is answered from the first that holds the file
  -p, --port <n>         TCP port, 0 picks a free one (default: ${kDefaultPort})
      --host <address>   address to bind (default: 127.0.0.1)
  -h, --help             show this
`;

const options = { directories: [], port: kDefaultPort, host: "127.0.0.1" };
for (let index = 2; index < process.argv.length; ++index) {
  const argument = process.argv[index];
  const next = () => process.argv[++index];
  if (argument === "-d" || argument === "--directory") options.directories.push(path.resolve(next()));
  else if (argument === "-p" || argument === "--port") options.port = Number(next());
  else if (argument === "--host") options.host = next();
  else if (argument === "-h" || argument === "--help") {
    process.stdout.write(usage);
    process.exit(0);
  } else {
    process.stderr.write(`unknown option: ${argument}\n${usage}`);
    process.exit(2);
  }
}
if (options.directories.length === 0) {
  options.directories.push(process.cwd());
}
for (const directory of options.directories) {
  if (!existsSync(directory) || !statSync(directory).isDirectory()) {
    process.stderr.write(`no such directory: ${directory}\n`);
    process.exit(1);
  }
}

const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Cache-Control": "no-store",
};

const handle = (request, response) => {
  const requested = decodeURIComponent(new URL(request.url, "http://placeholder").pathname);
  const relative = requested === "/" ? "index.html" : requested;
  const candidates = options.directories.map((root) => ({ root, file: path.join(root, relative) }));
  // a request must not escape a served directory, however it is spelled
  if (candidates.some(({ root, file }) => !file.startsWith(root + path.sep))) {
    response.writeHead(403, headers).end("forbidden");
    return;
  }
  const found = candidates.find(({ file }) => existsSync(file) && statSync(file).isFile());
  if (!found) {
    response.writeHead(404, headers).end("not found");
    if (requested !== "/favicon.ico") {
      process.stderr.write(`  404 ${requested}\n`);
    }
    return;
  }
  response.writeHead(200, {
    ...headers,
    "Content-Type": kContentTypes[path.extname(found.file).toLowerCase()] ?? "application/octet-stream",
    "Content-Length": statSync(found.file).size,
  });
  createReadStream(found.file).pipe(response);
};

const server = createServer(handle);
server.on("error", (error) => {
  process.stderr.write(`cannot listen on ${options.host}:${options.port}: ${error.message}\n`);
  process.exit(1);
});
server.listen(options.port, options.host, () => {
  // a test harness asks for port 0 and reads the chosen port from here
  process.stdout.write(`listening on port ${server.address().port}\n`);
  process.stdout.write(`serving ${options.directories.join(", ")} on http://localhost:${server.address().port}/ (COOP/COEP set)\n`);
});
