# Static Linking Exception

GNU Radio 4.0 is licensed under the GNU Lesser General Public License version 3 or later (LGPL-3.0-or-later)
with the following additional permission.

## What this means in practice

- **You may use GNU Radio 4.0 in proprietary applications** — including embedded systems, WASM builds,
  and statically linked executables — without disclosing your own source code.
- **Modifications to GNU Radio 4.0 itself** (the library source files) must be shared back under the same LGPL terms.
- **Your application code remains yours**, regardless of how it is linked (statically, dynamically, or compiled to WASM).

This exception exists because GNU Radio 4.0 targets platforms where dynamic linking is impractical or impossible
(e.g. embedded MCUs, WASM, safety-critical systems).
We want the library to be usable everywhere while ensuring that improvements to the library itself benefit everyone.

## Exception text

SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception

As a special exception to the GNU Lesser General Public License version 3 ("LGPL3"),
the copyright holders of this Library give you permission to convey to a third party a Combined Work
that links statically or dynamically to this Library without providing any Minimal Corresponding Source
or Minimal Application Code as set out in 4d or providing the installation information set out in section 4e,
provided that you comply with the other provisions of LGPL3,
and provided that you meet, for the Application, the terms and conditions of the license(s)
which apply to the Application.

Except as stated in this special exception, the provisions of LGPL3 will continue to apply in full to this Library.
If you modify this Library, you may apply this exception to your version of this Library,
but you are not obliged to do so. If you do not wish to do so, delete this exception statement from your version.
This exception does not (and cannot) modify any license terms which apply to the Application,
with which you must still comply.

## References

- [SPDX: LGPL-3.0-linking-exception](https://spdx.org/licenses/LGPL-3.0-linking-exception.html)
- [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.html)
- [Public Money? Public Code!](https://publiccode.eu/)
- [FAIR Principles](https://www.go-fair.org/fair-principles/)
