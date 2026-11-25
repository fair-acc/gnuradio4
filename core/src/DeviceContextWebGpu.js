// WebGPU side of DeviceContextWebGpu (`--js-library`). Runs on the browser main thread, which owns the GPUDevice;
// scheduler threads reach it through `emscripten_proxy_sync_with_ctx`, and each entry here finishes its proxying
// context once its promise settles. Kernels are the WGSL pipelines core/tools/wasm2wgsl wrote beside this module.
addToLibrary({
    $gr4WebGpu : {
        device : null,
        kernels : new Map(),   // table index -> {wgsl, stackBytes, refused}
        pipelines : new Map(), // table index -> Promise<GPUComputePipeline>
        heap : null,           // mirrors wasm memory [0, heap.size) at the same addresses
        staging : null,
        uniforms : null, // one 256-byte slot per launch of a flush
        error : "",
        staticEnd : 0,
        mirrorLimit : 0, // the adapter's limit on the GPU buffer, bytes
        generation : 0,  // bumped whenever `heap` is replaced, so a consumer rebuilds what binds it
        started : null,  // the promise of `gr4WebGpuStart`
        transferStaging : null,
        transferTemp : null,
        uniformWords : null,
        uploadStagings : [],                                        // {buffer, mapped}: MAP_WRITE buffers an upload to device-only memory is written into, each mapped again after its use
        bound : {heap : null, uniforms : null, groups : new Map()}, // table index -> a bind group per uniform slot, for these two buffers
    },

    // as DeviceContextWebGpu.cpp and the translator (WgslKernel.hpp) define them
    $gr4WebGpuConstants : {
        kDestinationDeviceOnly : 1, // a transfer's sides in device-only memory
        kSourceDeviceOnly : 2,
        kQueuedCopy : 0, // the `kernel` of a copy within device-only memory
        kMaxQueuedLaunches : 64,
        kWorkgroupSize : 64,
        kFault : 1, // `gr4WebGpuLaunch`'s status besides 0
        kRefusedByAdapter : 2,
    },

    // `gr4WebGpu[slot]`, made again in `step`-byte multiples when smaller than `bytes`; what it held is not kept
    $gr4WebGpuEnsureBuffer__deps : [ "$gr4WebGpu" ],
    $gr4WebGpuEnsureBuffer : (slot, bytes, step, usage) => {
        const state = gr4WebGpu;
        if (!state[slot] || state[slot].size < bytes) {
            state[slot]?.destroy();
            state[slot] = state.device.createBuffer({size : Math.max(step, Math.ceil(bytes / step) * step), usage});
        }
        return state[slot];
    },

    // the GPU buffer mirroring wasm memory [0, end) at the same addresses: it grows in 16 MiB steps keeping what the GPU
    // holds (device-only memory lives nowhere else), and a first one receives the static data, which kernels may read
    $gr4WebGpuEnsureHeap__deps : [ "$gr4WebGpu" ],
    $gr4WebGpuEnsureHeap : (end) => {
        const state            = gr4WebGpu;
        const device           = state.device;
        const kHeapGrowthBytes = 16 << 20;
        if (state.heap && state.heap.size >= end) {
            return;
        }
        const size  = Math.min(Math.ceil(Math.max(end, 4) / kHeapGrowthBytes) * kHeapGrowthBytes, state.mirrorLimit);
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX | GPUBufferUsage.INDIRECT; // a renderer may draw from it
        const grown = device.createBuffer({size, usage});
        if (state.heap) {
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(state.heap, 0, grown, 0, state.heap.size);
            device.queue.submit([ encoder.finish() ]);
            state.heap.destroy();
        } else {
            device.queue.writeBuffer(grown, 0, HEAPU8, 0, state.staticEnd & ~3);
        }
        state.heap = grown;
        state.generation += 1;
    },

    // a scratch buffer of at least `bytes` for a copy within the GPU buffer, kept for the next
    $gr4WebGpuTransferTemp__deps : [ "$gr4WebGpuEnsureBuffer" ],
    $gr4WebGpuTransferTemp : (bytes) => gr4WebGpuEnsureBuffer("transferTemp", bytes, 4096, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),

    // settles once `mapping` (a mapAsync) does: Chrome answers a map on its device tick, about 2 ms, unless another round
    // trip to its GPU process comes first, so one is kept going
    $gr4WebGpuPolled__deps : [ "$gr4WebGpu" ],
    $gr4WebGpuPolled : async (mapping) => {
        const device  = gr4WebGpu.device;
        let   settled = false;
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

    $gr4WebGpuMapPolled__deps : [ "$gr4WebGpuPolled" ],
    $gr4WebGpuMapPolled : (buffer, size) => gr4WebGpuPolled(buffer.mapAsync(GPUMapMode.READ, 0, size)),

    // bytes [lo, hi) of the GPU copy, both word-aligned
    $gr4WebGpuRead__deps : [ "$gr4WebGpu", "$gr4WebGpuEnsureBuffer", "$gr4WebGpuMapPolled" ],
    $gr4WebGpuRead : async (lo, hi) => {
        const state  = gr4WebGpu;
        const device = state.device;
        gr4WebGpuEnsureBuffer("transferStaging", hi - lo, 4096, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(state.heap, lo, state.transferStaging, 0, hi - lo);
        device.queue.submit([ encoder.finish() ]);
        await gr4WebGpuMapPolled(state.transferStaging, hi - lo);
        const bytes = new Uint8Array(state.transferStaging.getMappedRange(0, hi - lo)).slice();
        state.transferStaging.unmap();
        return bytes;
    },

    // copies `bytes` from `source` to `destination`, `sides` saying which is in device-only memory; an aligned copy within
    // it never comes here, as DeviceContextWebGpu.cpp queues it with the launches
    gr4WebGpuTransfer__deps : [ "$gr4WebGpu", "$gr4WebGpuConstants", "$gr4WebGpuEnsureHeap", "$gr4WebGpuRead", "$gr4WebGpuPolled", "emscripten_proxy_finish" ],
    gr4WebGpuTransfer : (ctx, destination, source, bytes, sides, statusPtr) => {
        const {kDestinationDeviceOnly, kSourceDeviceOnly, kFault} = gr4WebGpuConstants;
        const finish = (status) => {
            HEAP32[statusPtr >> 2] = status;
            _emscripten_proxy_finish(ctx);
        };
        (async () => {
            const state  = gr4WebGpu;
            const device = state.device;
            const words = (address) => [address & ~3, (address + bytes + 3) & ~3];
            gr4WebGpuEnsureHeap(Math.max(words(destination)[1], words(source)[1]));
            const aligned = (destination | source | bytes) % 4 === 0;
            if (sides === kDestinationDeviceOnly && aligned) {
                // a mapped buffer is shared memory with the GPU process, where writeBuffer copies through its command stream; two,
                // since one is mapped again only once the GPU copied its upload out, and with both pending the upload waits for one
                const kStagings                 = 2;
                const kUploadStagingGrowthBytes = 1 << 20;
                const mapped                    = state.uploadStagings.filter((staging) => staging.mapped);
                let   entry                     = mapped.find((staging) => staging.buffer.size >= bytes) ?? mapped[0];
                if (!entry && state.uploadStagings.length < kStagings) {
                    entry = {buffer : null, mapped : true, mapping : null};
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
                    entry.buffer = device.createBuffer({size : Math.ceil(bytes / kUploadStagingGrowthBytes) * kUploadStagingGrowthBytes, usage : GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation : true});
                }
                const staging = entry.buffer;
                new Uint8Array(staging.getMappedRange(0, bytes)).set(HEAPU8.subarray(source, source + bytes));
                staging.unmap();
                entry.mapped  = false;
                const encoder = device.createCommandEncoder();
                encoder.copyBufferToBuffer(staging, 0, state.heap, destination, bytes);
                device.queue.submit([ encoder.finish() ]);
                entry.mapping = staging.mapAsync(GPUMapMode.WRITE)
                                    .then(
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
            if ((sides & kSourceDeviceOnly) !== 0) {
                const [lo, hi] = words(source);
                data           = (await gr4WebGpuRead(lo, hi)).subarray(source - lo, source - lo + bytes);
            } else {
                data = HEAPU8.slice(source, source + bytes);
            }
            if ((sides & kDestinationDeviceOnly) === 0) {
                HEAPU8.set(data, destination);
            } else if (aligned) {
                device.queue.writeBuffer(state.heap, destination, data);
            } else { // patch the bytes into the words they share with their neighbours
                const [lo, hi] = words(destination);
                const span     = await gr4WebGpuRead(lo, hi);
                span.set(data, destination - lo);
                device.queue.writeBuffer(state.heap, lo, span);
            }
            finish(0);
        })().catch(() => finish(kFault));
    },

    // the device and the binary's kernels, once: an application may await `Module.gr4WebGpuStart()` at start-up, or hand
    // it a `GPUDevice` it created with the features and limits its renderer needs; the first call's answer stands, and
    // `registerWebGpuRuntime()` takes it. Resolves to {kind: 0 none, 1 software, 2 hardware, name}.
    $gr4WebGpuStart__deps : [ "$gr4WebGpu" ],
    $gr4WebGpuStart__postset : 'Module["gr4WebGpuStart"] = (device) => gr4WebGpuStart(device).then(() => gr4WebGpu.device);',
    $gr4WebGpuStart : (adopted) => (gr4WebGpu.started ??= (async () => {
        const describe = (info) => [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "unnamed adapter";
        const softwareIs = (info, text) => info.isFallbackAdapter === true || /swiftshader|llvmpipe|software/i.test(text);
        let   device                     = adopted;
        let   text                       = "the application's device";
        let   software                   = false;
        if (device) {
            const info = device.adapterInfo ?? {};
            text       = info.vendor ? describe(info) : text;
            software   = softwareIs(info, text);
        } else {
            if (!globalThis.navigator?.gpu) {
                return {kind : 0, name : "navigator.gpu is absent"};
            }
            // headless Chrome on NVIDIA sometimes answers the first request with null, a retried one not
            const kAdapterRequests = 3;
            const kAdapterRetryMs  = 250;
            let   adapter          = null;
            for (let attempt = 0; attempt < kAdapterRequests && !adapter; ++attempt) {
                if (attempt > 0) {
                    await new Promise((resolve) => setTimeout(resolve, kAdapterRetryMs));
                }
                adapter = await navigator.gpu.requestAdapter({powerPreference : "high-performance"});
            }
            if (!adapter) {
                return {kind : 0, name : `no WebGPU adapter after ${kAdapterRequests} requests, ${kAdapterRetryMs} ms apart`};
            }
            const info = adapter.info ?? {};
            text       = describe(info);
            software   = softwareIs(info, text);
            device     = await adapter.requestDevice({
                requiredLimits : {maxStorageBufferBindingSize : adapter.limits.maxStorageBufferBindingSize, maxBufferSize : adapter.limits.maxBufferSize},
                });
        }
        device.addEventListener("uncapturederror", (event) => (gr4WebGpu.error ||= event.error.message));
        device.lost.then((lost) => (gr4WebGpu.error ||= `device lost: ${lost.message}`));
        gr4WebGpu.device = device;
        // for a GPU consumer on this (main) thread, a renderer say: a device-only span's bytes lie in `heap` at its wasm
        // address, and work it submits runs after every flush submitted before; `heap` is null before the first launch;
        // `error` is the first validation error or device loss, latched ("" while none), and `device.lost` resolves on a loss
        Module["gr4WebGpuDevice"] = () => ({device : gr4WebGpu.device, heap : gr4WebGpu.heap, generation : gr4WebGpu.generation, error : gr4WebGpu.error});
        gr4WebGpu.mirrorLimit           = Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize, 0xffffffff) / 4) * 4;
        const url                       = _scriptName.replace(/\.js(\?.*)?$/, ".kernels.json");
        const response                  = await fetch(url, {cache : "no-cache"}); // a cached file may belong to an earlier build of the binary
        if (!response.ok) {
            return {kind : 0, name : `${url}: HTTP ${response.status}, so no kernel could run on the GPU`};
        }
        for (const kernel of (await response.json()).kernels ?? []) {
            const runnable = typeof kernel.wgsl === "string" && Number.isInteger(kernel.stackBytes) && kernel.stackBytes >= 0;
            if (Number.isInteger(kernel.table) && (runnable || typeof kernel.refused === "string")) {
                gr4WebGpu.kernels.set(kernel.table, kernel);
            }
        }
        return {kind : software ? 1 : 2, name : text};
    })().catch((error) => ({kind : 0, name : String(error)}))),

    gr4WebGpuInit__deps : [ "$gr4WebGpu", "$gr4WebGpuStart", "emscripten_proxy_finish" ],
    gr4WebGpuInit : (ctx, statusPtr, namePtr, nameBytes, staticEnd, mirrorLimitPtr) => {
        gr4WebGpu.staticEnd = staticEnd;
        gr4WebGpuStart().then(({kind, name}) => {
            HEAP32[statusPtr >> 2]       = kind;
            HEAPU32[mirrorLimitPtr >> 2] = gr4WebGpu.mirrorLimit;
            stringToUTF8(name, namePtr, nameBytes);
            _emscripten_proxy_finish(ctx);
        });
    },

    // bytes of stack per work item, or -1 when the binary carries no pipeline for this kernel, with the reason
    gr4WebGpuKernelStackBytes__deps : [ "$gr4WebGpu" ],
    gr4WebGpuKernelStackBytes__proxy : "sync",
    gr4WebGpuKernelStackBytes : (table, refusalPtr, refusalBytes) => {
        const kernel = gr4WebGpu.kernels.get(table);
        if (kernel && !kernel.refused) {
            return kernel.stackBytes;
        }
        stringToUTF8(kernel ? `not translated: ${kernel.refused}` : `table slot ${table} is not in the binary's .kernels.json`, refusalPtr, refusalBytes);
        return -1;
    },

    // batch: DeviceContextWebGpu.cpp's `FlushBatch`, each launch its `QueuedLaunch`. Runs the launches in one submit, in
    // order, then copies the download, if any, to wasm memory. Status 0: ran; kRefusedByAdapter: a lone launch the adapter
    // cannot build a pipeline for, so the CPU runs it, now and from then on; kFault: a device fault
    gr4WebGpuLaunch__deps : [ "$gr4WebGpu", "$gr4WebGpuConstants", "$gr4WebGpuEnsureBuffer", "$gr4WebGpuEnsureHeap", "$gr4WebGpuMapPolled", "$gr4WebGpuTransferTemp", "emscripten_proxy_finish" ],
    gr4WebGpuLaunch : (ctx, batch, messagePtr, messageBytes) => {
        const {kQueuedCopy, kMaxQueuedLaunches, kWorkgroupSize, kFault, kRefusedByAdapter}                                                       = gr4WebGpuConstants;
        const b                                                                                                                                  = batch >> 2;
        const [launchCount, launchesPtr, heapEnd, rangesPtr, rangeWords, livePtr, liveWords, downloadDestination, downloadSource, downloadBytes] = HEAPU32.subarray(b, b + 10);
        const launches                                                                                                                           = Array.from({length : launchCount}, (_, i) => {
            const [kernel, closure, count, stackTop, stackBytes] = HEAPU32.subarray((launchesPtr >> 2) + 5 * i, (launchesPtr >> 2) + 5 * i + 5);
            return {kernel, closure, count, stackTop, stackBytes};
                                                                                                                                  });
        const ranges                                                                                                                             = HEAPU32.slice(rangesPtr >> 2, (rangesPtr >> 2) + rangeWords); // [begin, end) pairs of live allocations
        const live                                                                                                                               = HEAPU32.slice(livePtr >> 2, (livePtr >> 2) + liveWords);      // the same unbridged: each lies in one of `ranges`
        const finish = (status, message = "") => {
            HEAP32[b + 10] = status;
            stringToUTF8(message, messagePtr, messageBytes);
            _emscripten_proxy_finish(ctx);
        };
        (async () => {
            const state     = gr4WebGpu;
            const device    = state.device;
            const pipelines = [];
            for (const {kernel} of launches) {
                if (kernel === kQueuedCopy) {
                    pipelines.push(null);
                    continue;
                }
                const translated = state.kernels.get(kernel);
                if (translated.refused) {
                    return finish(kRefusedByAdapter, translated.refused);
                }
                if (!state.pipelines.has(kernel)) {
                    // in a scope: a compile error would otherwise reach "uncapturederror" and fault every later launch
                    device.pushErrorScope("validation");
                    const module       = device.createShaderModule({code : translated.wgsl});
                    const compileError = await device.popErrorScope();
                    if (compileError) {
                        translated.refused = `the adapter could not compile it: ${compileError.message}`;
                        return finish(kRefusedByAdapter, translated.refused);
                    }
                    const pending = device.createComputePipelineAsync({layout : "auto", compute : {module, entryPoint : "main"}});
                    pending.catch(() => {}); // handled where it is awaited below; without this a rejection meanwhile counts as unhandled
                    state.pipelines.set(kernel, pending);
                }
                try {
                    pipelines.push(await state.pipelines.get(kernel));
                } catch (error) {
                    translated.refused = `the adapter could not build its pipeline: ${error}`;
                    return finish(kRefusedByAdapter, translated.refused);
                }
            }
            const kStagingGrowthBytes = 16 << 20;
            gr4WebGpuEnsureHeap(heapEnd);
            const readLo    = downloadSource & ~3;
            const readHi    = (downloadSource + downloadBytes + 3) & ~3;
            let   liveBytes = 4 + (downloadBytes > 0 ? readHi - readLo : 0);
            for (let r = 0; r < ranges.length; r += 2) {
                liveBytes += ranges[r + 1] - ranges[r];
            }
            gr4WebGpuEnsureBuffer("staging", liveBytes, kStagingGrowthBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
            // each live allocation as uploaded: after the dispatches only words the GPU changed are written back, so a host write
            // made meanwhile to another word survives; one to another byte of a word the GPU also changed does not
            const uploaded = [];
            for (let r = 0; r < ranges.length; r += 2) {
                const words = HEAPU32.slice(ranges[r] >> 2, ranges[r + 1] >> 2);
                uploaded.push(words);
                device.queue.writeBuffer(state.heap, ranges[r], words);
            }
            const kUniformSlot  = 256; // minUniformBufferOffsetAlignment: each launch reads its own slot
            const kUniformBytes = 32;  // WGSL's `Launch`, five u32 rounded up to the uniform alignment of 16
            gr4WebGpuEnsureBuffer("uniforms", launches.length * kUniformSlot, kMaxQueuedLaunches * kUniformSlot, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            if (launches.length > 0) {
                if (!state.uniformWords || state.uniformWords.length < state.uniforms.size / 4) {
                    state.uniformWords = new Uint32Array(state.uniforms.size / 4);
                }
                launches.forEach(({closure, count, stackTop, stackBytes}, i) => state.uniformWords.set([ closure, count, stackTop, stackBytes ], (i * kUniformSlot) / 4));
                device.queue.writeBuffer(state.uniforms, 0, state.uniformWords, 0, (launches.length * kUniformSlot) / 4);
            }

            // one per kernel and uniform slot, made once: creating them per dispatch costs the GPU process measurably
            if (state.bound.heap !== state.heap || state.bound.uniforms !== state.uniforms) {
                state.bound = {heap : state.heap, uniforms : state.uniforms, groups : new Map()};
            }
            device.pushErrorScope("validation");
            const encoder = device.createCommandEncoder();
            if (launches.length > 0) {
                const copied = gr4WebGpuTransferTemp(Math.max(0, ...launches.map(({kernel, count}) => (kernel === kQueuedCopy ? count : 0))));
                let   pass   = null;
                launches.forEach(({kernel, closure, count, stackTop}, i) => {
                    if (kernel === kQueuedCopy) { // through a scratch buffer: a copy within one buffer is not allowed
                        const [source, bytes, destination] = [ closure, count, stackTop ];
                        pass?.end();
                        pass = null;
                        encoder.copyBufferToBuffer(state.heap, source, copied, 0, bytes);
                        encoder.copyBufferToBuffer(copied, 0, state.heap, destination, bytes);
                        return;
                    }
                    const kMaxWorkgroupsPerDimension   = 65535;
                    pass                             ??= encoder.beginComputePass();
                    const groups                       = Math.ceil(count / kWorkgroupSize);
                    const rows                         = Math.ceil(groups / kMaxWorkgroupsPerDimension); // balanced, so no row is mostly idle
                    const width                        = Math.ceil(groups / rows);
                    if (!state.bound.groups.has(kernel)) {
                        state.bound.groups.set(kernel, []);
                    }
                    const slots   = state.bound.groups.get(kernel);
                    slots[i]    ??= device.createBindGroup({
                        layout : pipelines[i].getBindGroupLayout(0),
                        entries : [
                            {binding : 0, resource : {buffer : state.heap}},
                            {binding : 1, resource : {buffer : state.uniforms, offset : i * kUniformSlot, size : kUniformBytes}},
                        ],
                       });
                    pass.setPipeline(pipelines[i]);
                    pass.setBindGroup(0, slots[i]);
                    pass.dispatchWorkgroups(width, rows);
                });
                pass?.end();
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
            device.queue.submit([ encoder.finish() ]);
            const validation = await device.popErrorScope();
            if (validation) {
                return finish(kFault, validation.message);
            }
            await gr4WebGpuMapPolled(state.staging, Math.max(packed, 4));
            const mapped = state.staging.getMappedRange(0, Math.max(packed, 4));
            const result = new Uint32Array(mapped);
            for (let r = 0, offset = 0, l = 0; r < ranges.length; r += 2) { // a bridged gap's words are no allocation's
                const before = uploaded[r >> 1];
                const base   = ranges[r] >> 2;
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
            finish(state.error ? kFault : 0, state.error);
        })().catch((error) => finish(kFault, String(error)));
    },
});
