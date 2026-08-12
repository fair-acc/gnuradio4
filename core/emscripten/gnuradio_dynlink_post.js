// MAIN_MODULE exports __stack_pointer as a WebAssembly.Global, but SIDE_MODULE
// instantiation resolves env.__stack_pointer via wasmImports. Without this copy,
// dlopen fails: "imported mutable global must be a WebAssembly.Global object".
//
// GOT.func slots must hold table indices (numbers from addFunction), never raw JS
// functions. Writing a function into GOT.value makes call_indirect read a bad index
// → TypeError: getWasmTableEntry(...) is not a function.
//
// With ASSERTIONS=0, reportUndefinedSymbols can also throw on missing required
// symbols (typeof undefined.value). After the stock resolver runs, patch any
// leftover required slots with throwing stubs registered via addFunction.
//
// Linked automatically via gnuradio4::gnuradio-core-dynload (--post-js).
(function () {
  function grPublishStackPointerGlobal(exports) {
    var sp = exports && exports['__stack_pointer'];
    if (!sp && typeof ___stack_pointer !== 'undefined') {
      sp = ___stack_pointer;
    }
    if (sp && typeof wasmImports !== 'undefined') {
      wasmImports['__stack_pointer'] = sp;
    }
  }

  function grMissingStub(symName) {
    return function grUnresolvedGot() {
      throw new Error('[gnuradio4 dynlink] call to unresolved GOT symbol: ' + symName);
    };
  }

  function grGotNeedsFix(entry) {
    if (!entry) {
      return false;
    }
    // Unresolved sentinel, or a non-index left by a bad earlier patch.
    return entry.value === -1 || entry.value === 0 || typeof entry.value === 'function';
  }

  function grInstallGotValue(entry, value, symName) {
    if (typeof value === 'function') {
      if (typeof addFunction !== 'function') {
        console.error('[gnuradio4 dynlink] addFunction missing; cannot install', symName);
        return false;
      }
      // Prefer the symbol's declared sig when present (Emscripten JS library funcs).
      entry.value = value.sig ? addFunction(value, value.sig) : addFunction(value);
    } else {
      entry.value = value;
    }
    return true;
  }

  if (typeof assignWasmExports === 'function') {
    var grPreviousAssignWasmExports = assignWasmExports;
    assignWasmExports = function (exports) {
      grPreviousAssignWasmExports(exports);
      grPublishStackPointerGlobal(exports);
    };
  }

  if (typeof ___stack_pointer !== 'undefined') {
    grPublishStackPointerGlobal({'__stack_pointer': ___stack_pointer});
  }
  if (typeof wasmExports !== 'undefined') {
    grPublishStackPointerGlobal(wasmExports);
  }

  if (typeof reportUndefinedSymbols === 'function') {
    var grPreviousReportUndefinedSymbols = reportUndefinedSymbols;
    reportUndefinedSymbols = function () {
      try {
        grPreviousReportUndefinedSymbols();
      } catch (e) {
        console.error('[gnuradio4 dynlink] reportUndefinedSymbols failed:', e);
      }

      var missing = [];
      if (typeof GOT === 'undefined') {
        return;
      }
      for (var symName of Object.keys(GOT)) {
        var entry = GOT[symName];
        if (!grGotNeedsFix(entry)) {
          continue;
        }
        if (!entry.required && entry.value !== -1 && typeof entry.value !== 'function') {
          continue;
        }

        var resolved = null;
        if (typeof resolveGlobalSymbol === 'function') {
          resolved = resolveGlobalSymbol(symName, true).sym;
        }
        if (resolved && grInstallGotValue(entry, resolved, symName)) {
          continue;
        }
        if (entry.required || entry.value === -1 || typeof entry.value === 'function') {
          missing.push(symName);
          grInstallGotValue(entry, grMissingStub(symName), symName);
          entry.required = false;
        }
      }
      if (missing.length) {
        console.error('[gnuradio4 dynlink] unresolved GOT symbols (' + missing.length + '):', missing.slice(0, 30),
                      missing.length > 30 ? ('... +' + (missing.length - 30) + ' more') : '');
      }
    };
  }
})();
