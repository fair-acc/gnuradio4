// WebGPU side of DeviceContextWebGpu (`--js-library`). Runs on the browser main thread, which owns the GPUDevice;
// scheduler threads reach it through `emscripten_proxy_sync_with_ctx`, and each entry here finishes its proxying
// context once its promise settles. Kernels are the WGSL pipelines core/tools/wasm2wgsl wrote beside this module.
addToLibrary({
  $gr4WebGpu: {
    device: null,
    kernels: new Map(), // table index -> {wgsl, stackBytes, refused}
    pipelines: new Map(), // table index -> Promise<GPUComputePipeline>
    heap: null, // mirrors wasm memory [0, heap.size) at the same addresses
    staging: null,
    uniforms: null, // one 256-byte slot per launch of a flush
    error: "",
    slowPipelineReported: false,
    staticEnd: 0,
    transferStaging: null,
    transferTemp: null,
    uploadStagings: [], // {buffer, mapped}: MAP_WRITE buffers an upload to device-only memory is written into, each mapped again after its use
    bound: { heap: null, uniforms: null, groups: new Map() }, // table index -> a bind group per uniform slot, for these two buffers
  },

  // the GPU buffer mirroring wasm memory [0, end) at the same addresses: it grows in 16 MiB steps keeping what the GPU
  // holds (device-only memory lives nowhere else), and a first one receives the static data, which kernels may read
  $gr4WebGpuEnsureHeap__deps: ["$gr4WebGpu"],
  $gr4WebGpuEnsureHeap: (end) => {
    const state = gr4WebGpu;
    const device = state.device;
    const kStep = 16 << 20;
    if (state.heap && state.heap.size >= end) {
      return;
    }
    const grown = device.createBuffer({ size: Math.ceil(Math.max(end, 4) / kStep) * kStep, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (state.heap) {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(state.heap, 0, grown, 0, state.heap.size);
      device.queue.submit([encoder.finish()]);
      state.heap.destroy();
    } else {
      device.queue.writeBuffer(grown, 0, HEAPU8, 0, state.staticEnd & ~3);
    }
    state.heap = grown;
  },

  // settles once `mapping` (a mapAsync) does. Chrome answers a map on its device tick, about 2 ms, unless another round trip
  // to its GPU process comes first, so one is kept going: 16 dispatches and an 8-byte read-back took 2.3 ms with a plain
  // map and 0.18 ms with this (Radeon RX 7700S, Chrome 158)
  $gr4WebGpuPolled__deps: ["$gr4WebGpu"],
  $gr4WebGpuPolled: async (mapping) => {
    const device = gr4WebGpu.device;
    let settled = false;
    mapping.then(
      () => (settled = true),
      () => (settled = true),
    );
    while (!settled) {
      device.pushErrorScope("validation");
      await device.popErrorScope();
    }
    await mapping;
  },

  $gr4WebGpuMapPolled__deps: ["$gr4WebGpuPolled"],
  $gr4WebGpuMapPolled: (buffer, size) => gr4WebGpuPolled(buffer.mapAsync(GPUMapMode.READ, 0, size)),

  // bytes [lo, hi) of the GPU copy, both word-aligned
  $gr4WebGpuRead__deps: ["$gr4WebGpu", "$gr4WebGpuMapPolled"],
  $gr4WebGpuRead: async (lo, hi) => {
    const state = gr4WebGpu;
    const device = state.device;
    if (!state.transferStaging || state.transferStaging.size < hi - lo) {
      state.transferStaging?.destroy();
      state.transferStaging = device.createBuffer({ size: Math.ceil((hi - lo) / 4096) * 4096, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(state.heap, lo, state.transferStaging, 0, hi - lo);
    device.queue.submit([encoder.finish()]);
    await gr4WebGpuMapPolled(state.transferStaging, hi - lo);
    const bytes = new Uint8Array(state.transferStaging.getMappedRange(0, hi - lo)).slice();
    state.transferStaging.unmap();
    return bytes;
  },

  // copies `bytes` from `source` to `destination`; `sides` bit 0: the destination is device-only, bit 1: the source is
  gr4WebGpuTransfer__deps: ["$gr4WebGpu", "$gr4WebGpuEnsureHeap", "$gr4WebGpuRead", "$gr4WebGpuPolled", "emscripten_proxy_finish"],
  gr4WebGpuTransfer: (ctx, destination, source, bytes, sides, statusPtr) => {
    const finish = (status) => {
      HEAP32[statusPtr >> 2] = status;
      _emscripten_proxy_finish(ctx);
    };
    (async () => {
      const state = gr4WebGpu;
      const device = state.device;
      const words = (address) => [address & ~3, (address + bytes + 3) & ~3];
      gr4WebGpuEnsureHeap(Math.max(words(destination)[1], words(source)[1]));
      const aligned = (destination | source | bytes) % 4 === 0;
      if (sides === 3 && aligned) { // GPU to GPU through a scratch buffer: a copy within one buffer is not allowed
        if (!state.transferTemp || state.transferTemp.size < bytes) {
          state.transferTemp?.destroy();
          state.transferTemp = device.createBuffer({ size: Math.ceil(bytes / 4096) * 4096, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        }
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(state.heap, source, state.transferTemp, 0, bytes);
        encoder.copyBufferToBuffer(state.transferTemp, 0, state.heap, destination, bytes);
        device.queue.submit([encoder.finish()]);
        return finish(0);
      }
      if (sides === 1 && aligned) {
        // a mapped buffer is shared memory with the GPU process, where writeBuffer copies through its command stream:
        // one 512 KiB upload, 16 dispatches and a read-back took 0.30 ms against 0.46 ms (RTX 3070, Chrome 158, scratch spike.html)
        // two: a buffer is mapped again only once the GPU copied the upload out; with one, 1 to 2 in 10 uploads of
        // bm_FFT_backends came before that (RTX 3070). With both still being mapped again, the upload waits for one.
        const kStagings = 2;
        const kStep = 1 << 20;
        const mapped = state.uploadStagings.filter((staging) => staging.mapped);
        let entry = mapped.find((staging) => staging.buffer.size >= bytes) ?? mapped[0];
        if (!entry && state.uploadStagings.length < kStagings) {
          entry = { buffer: null, mapped: true, mapping: null };
          state.uploadStagings.push(entry);
        }
        if (!entry) {
          entry = state.uploadStagings.find((staging) => staging.buffer.size >= bytes) ?? state.uploadStagings[0];
          await gr4WebGpuPolled(entry.mapping);
        }
        if (!entry.mapped) { // the device is lost, which the next launch reports
          device.queue.writeBuffer(state.heap, destination, HEAPU8, source, bytes);
          return finish(0);
        }
        if (!entry.buffer || entry.buffer.size < bytes) {
          entry.buffer?.destroy();
          entry.buffer = device.createBuffer({ size: Math.ceil(bytes / kStep) * kStep, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
        }
        const staging = entry.buffer;
        new Uint8Array(staging.getMappedRange(0, bytes)).set(HEAPU8.subarray(source, source + bytes));
        staging.unmap();
        entry.mapped = false;
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(staging, 0, state.heap, destination, bytes);
        device.queue.submit([encoder.finish()]);
        entry.mapping = staging.mapAsync(GPUMapMode.WRITE).then(
          () => {
            if (entry.buffer === staging) {
              entry.mapped = true;
            }
          },
          () => {}, // a lost device: `mapped` stays false
        );
        return finish(0);
      }
      let data;
      if ((sides & 2) !== 0) {
        const [lo, hi] = words(source);
        data = (await gr4WebGpuRead(lo, hi)).subarray(source - lo, source - lo + bytes);
      } else {
        data = HEAPU8.slice(source, source + bytes);
      }
      if ((sides & 1) === 0) {
        HEAPU8.set(data, destination);
      } else if (aligned) {
        device.queue.writeBuffer(state.heap, destination, data);
      } else { // patch the bytes into the words they share with their neighbours
        const [lo, hi] = words(destination);
        const span = await gr4WebGpuRead(lo, hi);
        span.set(data, destination - lo);
        device.queue.writeBuffer(state.heap, lo, span);
      }
      finish(0);
    })().catch(() => finish(1));
  },

  gr4WebGpuInit__deps: ["$gr4WebGpu", "emscripten_proxy_finish"],
  gr4WebGpuInit: (ctx, statusPtr, namePtr, nameBytes, staticEnd) => {
    gr4WebGpu.staticEnd = staticEnd;
    const finish = (status, name) => {
      HEAP32[statusPtr >> 2] = status;
      stringToUTF8(name, namePtr, nameBytes);
      _emscripten_proxy_finish(ctx);
    };
    (async () => {
      if (!globalThis.navigator?.gpu) {
        return finish(0, "navigator.gpu is absent");
      }
      // headless Chrome on an NVIDIA RTX 3070 answered 5 of 113 first requests with null; both nulls retried got the adapter
      // on the second request, so up to three, 250 ms apart
      let adapter = null;
      for (let attempt = 0; attempt < 3 && !adapter; ++attempt) {
        if (attempt > 0) {
          console.log(`webgpu: no adapter on request ${attempt}, asking again in 250 ms`);
          await new Promise((resolve) => setTimeout(resolve, 250));
        }
        adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      }
      if (!adapter) {
        return finish(0, "no WebGPU adapter after three requests");
      }
      const info = adapter.info ?? {};
      const text = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "unnamed adapter";
      const software = info.isFallbackAdapter === true || /swiftshader|llvmpipe|software/i.test(text);
      const device = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
      });
      device.addEventListener("uncapturederror", (event) => (gr4WebGpu.error ||= event.error.message));
      device.lost.then((lost) => (gr4WebGpu.error ||= `device lost: ${lost.message}`));
      gr4WebGpu.device = device;
      const url = _scriptName.replace(/\.js(\?.*)?$/, ".kernels.json");
      const response = await fetch(url);
      if (response.ok) {
        for (const kernel of (await response.json()).kernels) {
          gr4WebGpu.kernels.set(kernel.table, kernel);
        }
      }
      console.log(`webgpu: ${software ? "software" : "hardware"} adapter '${text}', ${gr4WebGpu.kernels.size} kernels from ${url}`);
      finish(software ? 1 : 2, text);
    })().catch((error) => finish(0, String(error)));
  },

  // bytes of stack per work item, or -1 when the binary carries no pipeline for this kernel
  gr4WebGpuKernelStackBytes__deps: ["$gr4WebGpu"],
  gr4WebGpuKernelStackBytes__proxy: "sync",
  gr4WebGpuKernelStackBytes: (table) => {
    const kernel = gr4WebGpu.kernels.get(table);
    if (kernel?.refused) {
      console.log(`webgpu: kernel #${table} '${kernel.name}' runs on the CPU: ${kernel.refused}`);
    }
    return kernel && !kernel.refused ? kernel.stackBytes : -1;
  },

  // batch: u32 {launchCount, launches, staticEnd, heapEnd, ranges, rangeWords, live, liveWords, downloadDestination,
  // downloadSource, downloadBytes}, then i32 status; each launch u32 {kernel, closure, count, stackTop, stackBytes}. Runs the launches in
  // one submit, in order, then copies the download, if any, to wasm memory. Status 0: ran; 2: a lone launch the adapter
  // cannot build a pipeline for, so the CPU runs it, now and from then on; otherwise a device fault
  gr4WebGpuLaunch__deps: ["$gr4WebGpu", "$gr4WebGpuEnsureHeap", "$gr4WebGpuMapPolled", "emscripten_proxy_finish"],
  gr4WebGpuLaunch: (ctx, batch, messagePtr, messageBytes) => {
    const b = batch >> 2;
    const [launchCount, launchesPtr, staticEnd, heapEnd, rangesPtr, rangeWords, livePtr, liveWords, downloadDestination, downloadSource, downloadBytes] = HEAPU32.subarray(b, b + 11);
    const launches = Array.from({ length: launchCount }, (_, i) => {
      const [kernel, closure, count, stackTop, stackBytes] = HEAPU32.subarray((launchesPtr >> 2) + 5 * i, (launchesPtr >> 2) + 5 * i + 5);
      return { kernel, closure, count, stackTop, stackBytes };
    });
    const ranges = HEAPU32.slice(rangesPtr >> 2, (rangesPtr >> 2) + rangeWords); // [begin, end) pairs of live allocations
    const live = HEAPU32.slice(livePtr >> 2, (livePtr >> 2) + liveWords); // the same unbridged: each lies in one of `ranges`
    const finish = (status, message = "") => {
      HEAP32[b + 11] = status;
      stringToUTF8(message, messagePtr, messageBytes);
      _emscripten_proxy_finish(ctx);
    };
    (async () => {
      const state = gr4WebGpu;
      const device = state.device;
      const pipelines = [];
      for (const { kernel, count } of launches) {
        const translated = state.kernels.get(kernel);
        if (translated.refused) {
          return finish(2, translated.refused);
        }
        if (!state.pipelines.has(kernel)) {
          // in a scope: a compile error would otherwise reach "uncapturederror" and fault every later launch
          device.pushErrorScope("validation");
          const module = device.createShaderModule({ code: translated.wgsl });
          const compileError = await device.popErrorScope();
          if (compileError) {
            translated.refused = `the adapter could not compile it: ${compileError.message}`;
            console.log(`webgpu: kernel #${kernel} '${translated.name}' runs on the CPU: ${translated.refused}`);
            return finish(2, translated.refused);
          }
          const kSlowPipelineSeconds = 5; // NVIDIA builds a large pipeline for tens of seconds while its driver cache is cold
          const startedMs = performance.now();
          const pending = device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } }).then((built) => {
            const seconds = (performance.now() - startedMs) / 1000;
            if (seconds > kSlowPipelineSeconds && !state.slowPipelineReported) {
              state.slowPipelineReported = true;
              console.log(`webgpu: kernel #${kernel} '${translated.name}' waited ${seconds.toFixed(1)} s for its pipeline (${translated.wgsl.length} bytes of WGSL); a driver with a pipeline cache builds it faster next time; later slow pipelines are not reported`);
            }
            return built;
          });
          pending.catch(() => {}); // handled where it is awaited below; without this a rejection meanwhile counts as unhandled
          state.pipelines.set(kernel, pending);
          console.log(`webgpu: first dispatch of kernel #${kernel} (${count} work items)`);
        }
        try {
          pipelines.push(await state.pipelines.get(kernel));
        } catch (error) {
          translated.refused = `the adapter could not build its pipeline: ${error}`;
          console.log(`webgpu: kernel #${kernel} '${translated.name}' runs on the CPU: ${translated.refused}`);
          return finish(2, translated.refused);
        }
      }
      const kStep = 16 << 20;
      gr4WebGpuEnsureHeap(heapEnd);
      const readLo = downloadSource & ~3;
      const readHi = (downloadSource + downloadBytes + 3) & ~3;
      let liveBytes = 4 + (downloadBytes > 0 ? readHi - readLo : 0);
      for (let r = 0; r < ranges.length; r += 2) {
        liveBytes += ranges[r + 1] - ranges[r];
      }
      if (!state.staging || state.staging.size < liveBytes) {
        state.staging?.destroy();
        state.staging = device.createBuffer({ size: Math.ceil(liveBytes / kStep) * kStep, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      }
      // each live allocation as uploaded: after the dispatches only words the GPU changed are written back, so a host write
      // made meanwhile to another word survives; one to another byte of a word the GPU also changed does not
      const uploaded = [];
      for (let r = 0; r < ranges.length; r += 2) {
        const words = HEAPU32.slice(ranges[r] >> 2, ranges[r + 1] >> 2);
        uploaded.push(words);
        device.queue.writeBuffer(state.heap, ranges[r], words);
      }
      const kUniformSlot = 256; // minUniformBufferOffsetAlignment: each launch reads its own slot
      if (!state.uniforms || state.uniforms.size < launches.length * kUniformSlot) {
        state.uniforms?.destroy();
        state.uniforms = device.createBuffer({ size: Math.max(64, launches.length) * kUniformSlot, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      }
      if (launches.length > 0) {
        const uniforms = new Uint32Array((launches.length * kUniformSlot) / 4);
        launches.forEach(({ closure, count, stackTop, stackBytes }, i) => uniforms.set([closure, count, stackTop, stackBytes], (i * kUniformSlot) / 4));
        device.queue.writeBuffer(state.uniforms, 0, uniforms);
      }

      // made once: a bind group created per dispatch cost 0.06 ms per flush of 16 (RTX 3070, Chrome 158, scratch flush.html)
      if (state.bound.heap !== state.heap || state.bound.uniforms !== state.uniforms) {
        state.bound = { heap: state.heap, uniforms: state.uniforms, groups: new Map() };
      }
      device.pushErrorScope("validation");
      const encoder = device.createCommandEncoder();
      if (launches.length > 0) {
        const pass = encoder.beginComputePass();
        launches.forEach(({ kernel, count }, i) => {
          const groups = Math.ceil(count / 64);
          const rows = Math.ceil(groups / 65535); // maxComputeWorkgroupsPerDimension; balanced, so no row is mostly idle
          const width = Math.ceil(groups / rows);
          if (!state.bound.groups.has(kernel)) {
            state.bound.groups.set(kernel, []);
          }
          const slots = state.bound.groups.get(kernel);
          slots[i] ??= device.createBindGroup({
            layout: pipelines[i].getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: state.heap } },
              { binding: 1, resource: { buffer: state.uniforms, offset: i * kUniformSlot, size: 32 } },
            ],
          });
          pass.setPipeline(pipelines[i]);
          pass.setBindGroup(0, slots[i]);
          pass.dispatchWorkgroups(width, rows);
        });
        pass.end();
      }
      let packed = 0;
      for (let r = 0; r < ranges.length; r += 2) {
        encoder.copyBufferToBuffer(state.heap, ranges[r], state.staging, packed, ranges[r + 1] - ranges[r]);
        packed += ranges[r + 1] - ranges[r];
      }
      const readAt = packed;
      if (downloadBytes > 0) {
        encoder.copyBufferToBuffer(state.heap, readLo, state.staging, readAt, readHi - readLo);
        packed += readHi - readLo;
      }
      device.queue.submit([encoder.finish()]);
      const validation = await device.popErrorScope();
      if (validation) {
        return finish(1, validation.message);
      }
      await gr4WebGpuMapPolled(state.staging, Math.max(packed, 4));
      const mapped = state.staging.getMappedRange(0, Math.max(packed, 4));
      const result = new Uint32Array(mapped);
      for (let r = 0, offset = 0, l = 0; r < ranges.length; r += 2) { // a bridged gap's words are no allocation's
        const before = uploaded[r >> 1];
        const base = ranges[r] >> 2;
        for (; l < live.length && live[l] < ranges[r + 1]; l += 2) {
          for (let w = (live[l] >> 2) - base; w < (live[l + 1] >> 2) - base; ++w) {
            if (result[offset + w] !== before[w]) {
              HEAPU32[base + w] = result[offset + w];
            }
          }
        }
        offset += before.length;
      }
      if (downloadBytes > 0) {
        HEAPU8.set(new Uint8Array(mapped, readAt + (downloadSource - readLo), downloadBytes), downloadDestination);
      }
      state.staging.unmap();
      finish(state.error ? 1 : 0, state.error);
    })().catch((error) => finish(1, String(error)));
  },
});
