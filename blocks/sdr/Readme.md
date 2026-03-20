# RTL2832Source — SDR blocks

## TL;DR

an `RTL2832Source<T>` block that talks directly to RTL2832U + R820T/R820T2/R860/E4000
hardware over USB. No SoapySDR, no librtlsdr, no libusb — just Linux USB
ioctls (`/dev/bus/usb`, zero external dependencies), with a WebUSB path for the browser.

Output types: `std::uint8_t` (raw interleaved I/Q) or `std::complex<float>`
(normalised to ±1).

For other SDR hardware or macOS/Windows native support, GR4 also provides a `SoapySdrSource` block
via [SoapySDR](https://github.com/pothosware/SoapySDR)(Boost-1.0))) — though for RTL-SDR specifically, even that path
links against librtlsdr (GPLv2+),
making the direct implementation here the only fully permissive native route to the hardware.

License: MIT "(see License section for the `-DMIT_ONLY=ON` opt-out)"

## What it does

Each RTL2832U-based SDR dongle consists of two independent chips that are wired together:
the RTL2832U demodulator and a tuner (R820T/R820T2/R860 or E4000), programmed over I2C/USB using
register-level protocols reimplemented in C++ (native) and JavaScript (WASM).

This is a clean reimplementation derived from [jtarrio/webrtlsdr](https://github.com/jtarrio/webrtlsdr) (Apache-2.0), itself a continuation of [google/radioreceiver](https://github.com/google/radioreceiver) (Apache-2.0, Google, 2013)
— both by [Jacobo Tarrío](https://github.com/jtarrio), the original author.
Additional ideas are from [Sandeep Mistry](https://github.com/sandeepmistry) ([rtlsdrjs](https://github.com/sandeepmistry/rtlsdrjs)), [Maximising USB bulk throughput](https://threephase.xyz/2012/02/bulk-usb-throughput),
and [Raspberry Pi Pico WebUSB performance](https://suyashsingh.in/blog/raspberrypi-pico-webusb-performance-test).

This implementation extends that work with multi-in-flight pipelined bulk transfers
(12 concurrent vs. sequential), a lock-free SPSC queue over SharedArrayBuffer
for the WASM path, and integration with the GR4 block lifecycle and scheduler.

## Register documentation

The RTL2832U and R820T/R820T2 register protocols used here are documented in
datasheets that were originally covered by NDAs but have since become publicly available and widely used by the SDR community for over a decade:

| Document                                                                                                           | Origin             | Public since                                                                              |
| ------------------------------------------------------------------------------------------------------------------ | ------------------ | ----------------------------------------------------------------------------------------- |
| [RTL2832U Datasheet (Rev 1.4)](https://homepages.uni-regensburg.de/~erc24492/SDR/Data_rtl2832u.pdf)                | Realtek, 2010      | ~2012                                                                                     |
| [R820T Datasheet](https://www.rtl-sdr.com/wp-content/uploads/2013/04/R820T_datasheet-Non_R-20111130_unlocked1.pdf) | Rafael Micro, 2011 | Sept 2012                                                                                 |
| [R820T2 Register Description](https://www.rtl-sdr.com/wp-content/uploads/2016/12/R820T2_Register_Description.pdf)  | Rafael Micro, 2012 | [Dec 2016](https://www.rtl-sdr.com/r820t2-register-description-data-sheet-now-available/) |

The R860 tuner found in recent dongles is electrically identical to the R820T2.

## License

MIT — same as the rest of GR4.

The CMake flag `-DMIT_ONLY=ON` suppresses this block entirely.

While there is no legal obligation to do so, it is there if you want to honour
the years of reverse-engineering by the Osmocom community that produced the
register-level knowledge all RTL-SDR software builds on. Their library work is
GPLv2+-licensed and we respect that choice. This implementation is not based
on the Osmocom code but on the [webrtlsdr](https://github.com/jtarrio/webrtlsdr)
TypeScript implementation (Apache-2.0).

GR4 targets browser and embedded deployment where (L)GPL distribution obligations
can be difficult to satisfy in practice. MIT keeps the code usable everywhere GR4 runs.

## Acknowledgements

The ability to use cheap DVB-T dongles as software-defined radios exists because of
the [Osmocom RTL-SDR project](https://osmocom.org/projects/rtl-sdr/wiki)
and a remarkable community effort spanning 15 years.

The register-level knowledge that _every_ RTL-SDR implementation depends on
— including this one — originates from their reverse-engineering work.
If you enjoy RTL-SDR, or want to support their work, please
consider [donating to Osmocom](https://opencollective.com/osmocom).

Special thanks goes to: [Eric Fry](https://osmocom.org/projects/rtl-sdr/wiki) (USB packet reverse-engineering, ~2010),
[Antti Palosaari](https://github.com/crope) (Linux Kernel DVB/SDR drivers, raw I/Q streaming, 2012),
[Steve Markgraf](https://github.com/steve-m) (librtlsdr author, R820T support), Dimitri Stolnikov (gr-osmosdr),
Hoernchen, [Harald Welte](https://github.com/laf0rge) (Osmocom founder), Christian Vogel,
and other authors listed on the Osmocom project page.

Further key contributions that shaped the community's understanding of the R820T/R820T2:
[Kyle Keen / keenerd](https://github.com/keenerd) (`rtl_fm`, `rtl_power`, direct sampling), [tejeez](https://github.com/tejeez) (no-mod HF reception, fast retuning),
[Oliver Jowett](https://github.com/mutability) (extended tuning range, PLL precision), [Hayati Ayguen](https://github.com/hayguen) ([librtlsdr fork](https://github.com/librtlsdr/librtlsdr) — IF bandwidth filters, sideband selection),
[Carl Laufer / RTL-SDR Blog](https://github.com/rtlsdrblog) ([V3/V4 driver fork](https://github.com/rtlsdrblog/rtl-sdr-blog)), Luigi Tarenga (obtained and published the [R820T2 Register Description](https://www.rtl-sdr.com/r820t2-register-description-data-sheet-now-available/), 2016).
