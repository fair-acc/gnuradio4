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

## What stays in force

The exception waives only the relinking requirement in LGPLv3 section 4.
Every other LGPLv3 provision remains in full effect, including:

- **Modifications to the library source** must be shared back under the
  same terms.
- **Patent grant (LGPLv3 §11).** Each contributor, by contributing their
  code, gives everyone using GR4 — including commercial users — a
  written promise that they will not later sue over patents covering
  the code they contributed. The promise stays with the patent if the
  contributor's company is later acquired, merges, or transfers the
  patent, and cannot be selectively withheld from some users while
  extended to others.
- **Defensive termination of the copyright license** with a 30/60-day
  cure window for first-time violations (LGPLv3 / GPLv3 §8).

This combination is what makes GR4 simultaneously easy to use in
proprietary, embedded, and WASM contexts *and* safe for downstream users
— including commercial ones — to build products on top of.

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
