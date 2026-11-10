#!/usr/bin/env node
// Runs a WebAssembly test page in headless Chrome and exits with the code the page publishes as globalThis.gr4TestExitCode.
// usage: run_wasm_test.mjs <directory> <page>
//   run_wasm_test.mjs build/core/test "wasm_test.html?test=qa_CRC"   an Emscripten qa_ test, through wasm_test.html beside this script
//   run_wasm_test.mjs build/blocks/audio/test "WebAudioIntegrationProbe.html?auto"   a page of its own
// Exits 77 (ctest's SKIP_RETURN_CODE) when no browser is installed.

import { NoBrowserError, openPage, withBrowser } from "./browser_harness.mjs";

const kSkipped = 77;
const kTestDeadlineMs = 1_400_000; // below ctest's default timeout of 1500 s, so a hung page is reported here, with its output

const [directory, page] = process.argv.slice(2);
if (!directory || !page) {
  process.stderr.write("usage: run_wasm_test.mjs <directory> <page>\n");
  process.exit(2);
}

let exitCode = 1;
try {
  exitCode = await withBrowser([directory, import.meta.dirname], async ({ server, browser }) => {
    const tab = await openPage(browser, `${server.origin}/${page}`);
    try {
      const published = await tab.waitFor("globalThis.gr4TestExitCode === undefined ? null : String(globalThis.gr4TestExitCode)", kTestDeadlineMs);
      return Number(published);
    } finally {
      process.stdout.write(tab.consoleLines.map((line) => `${line}\n`).join(""));
    }
  });
} catch (error) {
  if (error instanceof NoBrowserError) {
    process.stdout.write(`skipped: ${error.message}\n`);
    process.exit(kSkipped);
  }
  process.stderr.write(`${error.stack ?? error}\n`);
}
process.exit(exitCode);
