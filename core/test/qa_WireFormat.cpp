#include <boost/ut.hpp>

#include <gnuradio-4.0/CRC.hpp>
#include <gnuradio-4.0/WireFormat.hpp>
#include <gnuradio-4.0/WireStream.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::byte> makeBuffer(std::size_t n) {
    std::vector<std::byte> v(n);
    std::uint32_t          s = 0x9E3779B9U;
    for (std::size_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v[i] = static_cast<std::byte>(s & 0xFFU);
    }
    return v;
}

std::span<const std::byte> asSpan(const std::vector<std::byte>& v) noexcept { return std::span<const std::byte>(v.data(), v.size()); }

} // namespace

const boost::ut::suite<"WireFormat::Annotation"> _annotation_tests = [] {
    using namespace boost::ut;
    using gr::wire::Annotation;
    using gr::wire::annotationByteSize;
    using gr::wire::readAnnotation;
    using gr::wire::writeAnnotation;

    "round-trip all three strings"_test = [] {
        Annotation             a{.quantity = "frequency", .unit = "Hz", .description = "carrier frequency of the channel"};
        std::vector<std::byte> buf(annotationByteSize(a));
        const auto             written = writeAnnotation(buf, a);
        expect(written.has_value());
        expect(eq(*written, annotationByteSize(a)));
        const auto r = readAnnotation(asSpan(buf));
        expect(r.has_value());
        expect(eq(r->bytesRead, *written));
        expect(eq(r->a.quantity, a.quantity));
        expect(eq(r->a.unit, a.unit));
        expect(eq(r->a.description, a.description));
    };

    "round-trip with empty fields — independently absent via len == 0"_test = [] {
        for (Annotation a : {Annotation{}, Annotation{.quantity = "voltage"}, Annotation{.unit = "V"}, Annotation{.description = "battery cell"}, Annotation{.quantity = "frequency", .description = "no unit known"}}) {
            std::vector<std::byte> buf(annotationByteSize(a));
            expect(writeAnnotation(buf, a).has_value());
            const auto r = readAnnotation(asSpan(buf));
            expect(r.has_value());
            expect(eq(r->a.quantity, a.quantity));
            expect(eq(r->a.unit, a.unit));
            expect(eq(r->a.description, a.description));
        }
    };

    "UTF-8 round-trip — Greek letters and SI special chars"_test = [] {
        Annotation             a{.quantity = "angular frequency", .unit = "rad/s", .description = "ω = 2π·f (Ω at 25 °C)"};
        std::vector<std::byte> buf(annotationByteSize(a));
        expect(writeAnnotation(buf, a).has_value());
        const auto r = readAnnotation(asSpan(buf));
        expect(r.has_value());
        expect(eq(r->a.unit, a.unit));
        expect(eq(r->a.description, a.description));
    };

    "u8 max boundary for quantity and unit (255 bytes each)"_test = [] {
        const std::string      bigQ(255UZ, 'q');
        const std::string      bigU(255UZ, 'u');
        Annotation             a{.quantity = bigQ, .unit = bigU, .description = "d"};
        std::vector<std::byte> buf(annotationByteSize(a));
        expect(writeAnnotation(buf, a).has_value());
        const auto r = readAnnotation(asSpan(buf));
        expect(r.has_value());
        expect(eq(r->a.quantity.size(), std::size_t{255UZ}));
        expect(eq(r->a.unit.size(), std::size_t{255UZ}));
    };

    "max-length description (just under u16 max)"_test = [] {
        const std::string      bigDesc(65535UZ, 'x');
        Annotation             a{.quantity = "q", .unit = "u", .description = bigDesc};
        std::vector<std::byte> buf(annotationByteSize(a));
        expect(writeAnnotation(buf, a).has_value());
        const auto r = readAnnotation(asSpan(buf));
        expect(r.has_value());
        expect(eq(r->a.description.size(), std::size_t{65535UZ}));
        expect(eq(r->a.description.front(), 'x'));
        expect(eq(r->a.description.back(), 'x'));
    };

    "empty input → nullopt"_test = [] {
        const auto r = readAnnotation(std::span<const std::byte>{});
        expect(not r.has_value());
    };

    "truncated input rejected at every cut point"_test = [] {
        Annotation             a{.quantity = "frequency", .unit = "Hz", .description = "carrier"};
        std::vector<std::byte> buf(annotationByteSize(a));
        expect(writeAnnotation(buf, a).has_value());
        for (std::size_t cut = 0UZ; cut < buf.size(); ++cut) {
            const auto r = readAnnotation(std::span<const std::byte>(buf.data(), cut));
            expect(not r.has_value()) << "cut = " << cut;
        }
        const auto rOk = readAnnotation(asSpan(buf));
        expect(rOk.has_value()); // full buffer succeeds
    };

    "writeAnnotation rejects under-sized destination"_test = [] {
        Annotation             a{.quantity = "frequency", .unit = "Hz", .description = "carrier"};
        std::vector<std::byte> tooSmall(annotationByteSize(a) - 1UZ);
        expect(not writeAnnotation(tooSmall, a).has_value());
    };

    "writeAnnotation rejects oversize fields (silent narrowing guard)"_test = [] {
        const std::string      tooLong(256UZ, 'q');
        Annotation             a{.quantity = tooLong, .unit = "Hz", .description = ""};
        std::vector<std::byte> buf(annotationByteSize(a));
        expect(not writeAnnotation(buf, a).has_value()) << "256-byte quantity must be rejected";

        const std::string      farTooLong(65536UZ, 'd');
        Annotation             a2{.quantity = "q", .unit = "Hz", .description = farTooLong};
        std::vector<std::byte> buf2(annotationByteSize(a2));
        expect(not writeAnnotation(buf2, a2).has_value()) << "65 536-byte description must be rejected";
    };
};

const boost::ut::suite<"WireFormat::StreamHeader"> _stream_header_tests = [] {
    using namespace boost::ut;
    using gr::wire::kStreamFlagFrameCRC;
    using gr::wire::kStreamHeaderBytes;
    using gr::wire::kStreamVersionV1;
    using gr::wire::readStreamHeader;
    using gr::wire::StreamHeader;
    using gr::wire::writeStreamHeader;

    "magic + version + flags round-trip"_test = [] {
        std::array<std::byte, kStreamHeaderBytes> buf{};
        const StreamHeader                        in{.flags = kStreamFlagFrameCRC};
        writeStreamHeader(buf, in);
        const auto out = readStreamHeader(buf);
        expect(out.has_value());
        expect(eq(out->version, kStreamVersionV1));
        expect(eq(out->flags, kStreamFlagFrameCRC));
        expect(out->magic == in.magic);
    };

    "bad magic rejected"_test = [] {
        std::array<std::byte, kStreamHeaderBytes> buf{};
        writeStreamHeader(buf, StreamHeader{});
        buf[0]         = std::byte{'X'};
        const auto out = readStreamHeader(buf);
        expect(not out.has_value());
    };

    "unknown version rejected"_test = [] {
        std::array<std::byte, kStreamHeaderBytes> buf{};
        writeStreamHeader(buf, StreamHeader{});
        const std::uint16_t v = 0x9999U;
        std::memcpy(buf.data() + 4, &v, sizeof(v));
        const auto out = readStreamHeader(buf);
        expect(not out.has_value());
    };

    "truncated buffer rejected"_test = [] {
        std::array<std::byte, kStreamHeaderBytes> buf{};
        writeStreamHeader(buf, StreamHeader{});
        const auto out = readStreamHeader(std::span<const std::byte>(buf.data(), 4));
        expect(not out.has_value());
    };
};

const boost::ut::suite<"WireFormat::FrameEnvelope"> _frame_envelope_tests = [] {
    using namespace boost::ut;
    using gr::wire::computeFrameCrc;
    using gr::wire::FrameEnvelope;
    using gr::wire::kFrameCrcOffset;
    using gr::wire::kFrameEnvelopeBytes;
    using gr::wire::readFrameEnvelope;
    using gr::wire::writeFrameEnvelope;

    "envelope round-trip — no CRC"_test = [] {
        std::array<std::byte, kFrameEnvelopeBytes> buf{};
        const FrameEnvelope                        in{.frame_size = 4321U, .sequence = 99ULL, .crc32c = 0U};
        writeFrameEnvelope(buf, in);
        const auto out = readFrameEnvelope(buf);
        expect(out.has_value());
        expect(eq(out->frame_size, in.frame_size));
        expect(eq(out->sequence, in.sequence));
        expect(eq(out->crc32c, in.crc32c));
    };

    "CRC zero-out trick — happy path"_test = [] {
        const auto                                 payload = makeBuffer(1024UZ);
        std::array<std::byte, kFrameEnvelopeBytes> envBuf{};
        FrameEnvelope                              env{.frame_size = static_cast<std::uint32_t>(payload.size()), .sequence = 7ULL, .crc32c = 0U};
        writeFrameEnvelope(envBuf, env);
        const auto crc = computeFrameCrc(envBuf, asSpan(payload));

        env.crc32c = crc;
        writeFrameEnvelope(envBuf, env);

        std::array<std::byte, kFrameEnvelopeBytes> envCopy = envBuf;
        std::memset(envCopy.data() + kFrameCrcOffset, 0, sizeof(std::uint32_t));
        expect(eq(crc, computeFrameCrc(envCopy, asSpan(payload))));
    };

    "CRC catches a single-bit flip in payload"_test = [] {
        auto                                       payload = makeBuffer(256UZ);
        std::array<std::byte, kFrameEnvelopeBytes> envBuf{};
        FrameEnvelope                              env{.frame_size = static_cast<std::uint32_t>(payload.size()), .sequence = 0ULL, .crc32c = 0U};
        writeFrameEnvelope(envBuf, env);
        const auto good = computeFrameCrc(envBuf, asSpan(payload));
        payload[128]    = static_cast<std::byte>(static_cast<std::uint8_t>(payload[128]) ^ 0x01U);
        const auto bad  = computeFrameCrc(envBuf, asSpan(payload));
        expect(not eq(good, bad));
    };
};

const boost::ut::suite<"WireFormat::StreamReaderWriter"> _stream_rw_tests = [] {
    using namespace boost::ut;
    using gr::wire::FrameStatus;
    using gr::wire::kFrameEnvelopeBytes;
    using gr::wire::kStreamFlagFrameCRC;
    using gr::wire::kStreamHeaderBytes;
    using gr::wire::WireStreamReader;
    using gr::wire::WireStreamWriter;

    "end-to-end: header + N frames, with frame CRC"_test = [] {
        WireStreamWriter       w(kStreamFlagFrameCRC);
        std::vector<std::byte> out;
        w.writeHeader(out);

        const std::vector<std::vector<std::byte>> payloads = {makeBuffer(16UZ), makeBuffer(1024UZ), makeBuffer(4096UZ), makeBuffer(127UZ)};
        for (const auto& p : payloads) {
            w.writeFrame(out, asSpan(p));
        }

        WireStreamReader r;
        const auto       header = r.begin(out);
        expect(header.has_value());
        expect(eq(header->flags, kStreamFlagFrameCRC));

        for (std::size_t i = 0; i < payloads.size(); ++i) {
            const auto fv = r.next();
            expect(fv.has_value());
            expect(fv->status == FrameStatus::Ok);
            expect(eq(fv->envelope.sequence, std::uint64_t{i}));
            expect(eq(fv->payload.size(), payloads[i].size()));
            expect(std::equal(fv->payload.begin(), fv->payload.end(), payloads[i].begin()));
        }
        expect(not r.next().has_value());
    };

    "end-to-end: header + N frames, without frame CRC"_test = [] {
        WireStreamWriter       w(0U);
        std::vector<std::byte> out;
        w.writeHeader(out);
        const std::vector<std::vector<std::byte>> payloads = {makeBuffer(8UZ), makeBuffer(64UZ)};
        for (const auto& p : payloads) {
            w.writeFrame(out, asSpan(p));
        }

        WireStreamReader r;
        const auto       header = r.begin(out);
        expect(header.has_value());
        expect(eq(header->flags, std::uint16_t{0U}));
        for (std::size_t i = 0; i < payloads.size(); ++i) {
            const auto fv = r.next();
            expect(fv.has_value());
            expect(fv->status == FrameStatus::Ok);
            expect(eq(fv->envelope.sequence, std::uint64_t{i}));
        }
    };

    "empty-payload frame round-trips"_test = [] {
        WireStreamWriter       w(0U);
        std::vector<std::byte> out;
        w.writeHeader(out);
        w.writeFrame(out, std::span<const std::byte>{});

        WireStreamReader r;
        const auto       h = r.begin(out);
        expect(h.has_value());
        const auto fv = r.next();
        expect(fv.has_value());
        expect(fv->status == FrameStatus::Ok);
        expect(eq(fv->envelope.frame_size, std::uint32_t{0U}));
        expect(eq(fv->payload.size(), std::size_t{0UZ}));
    };

    "writeFrame auto-emits stream header if not yet written"_test = [] {
        WireStreamWriter       w(kStreamFlagFrameCRC);
        std::vector<std::byte> out;
        const auto             payload = makeBuffer(32UZ);
        w.writeFrame(out, asSpan(payload)); // no explicit writeHeader

        WireStreamReader r;
        const auto       h = r.begin(out);
        expect(h.has_value());
        expect(eq(h->flags, kStreamFlagFrameCRC)); // auto-emit uses constructor flags
        const auto fv = r.next();
        expect(fv.has_value());
        expect(fv->status == FrameStatus::Ok);
        expect(eq(fv->envelope.sequence, std::uint64_t{0U}));
    };

    "reader flags BadCrc when payload tampered post-write"_test = [] {
        WireStreamWriter       w(kStreamFlagFrameCRC);
        std::vector<std::byte> out;
        w.writeHeader(out);
        const auto payload = makeBuffer(512UZ);
        w.writeFrame(out, asSpan(payload));

        const std::size_t tamperPos = kStreamHeaderBytes + kFrameEnvelopeBytes + 50UZ;
        out[tamperPos]              = static_cast<std::byte>(static_cast<std::uint8_t>(out[tamperPos]) ^ 0x80U);

        WireStreamReader r;
        const auto       h = r.begin(out);
        expect(h.has_value());
        const auto fv = r.next();
        expect(fv.has_value());
        expect(fv->status == FrameStatus::BadCrc);
    };

    "reader flags BadSequence when frame sequence tampered"_test = [] {
        WireStreamWriter       w(0U); // no CRC — isolate the sequence check
        std::vector<std::byte> out;
        w.writeHeader(out);
        const auto payload = makeBuffer(64UZ);
        w.writeFrame(out, asSpan(payload));
        w.writeFrame(out, asSpan(payload));

        // tamper the 2nd frame's sequence field (offset 4 within its envelope)
        const std::size_t   secondEnvPos = kStreamHeaderBytes + kFrameEnvelopeBytes + payload.size();
        const std::uint64_t tamperSeq    = 999ULL;
        std::memcpy(out.data() + secondEnvPos + 4, &tamperSeq, sizeof(tamperSeq));

        WireStreamReader r;
        const auto       h = r.begin(out);
        expect(h.has_value());
        const auto fv1 = r.next();
        expect(fv1.has_value());
        expect(fv1->status == FrameStatus::Ok);
        const auto fv2 = r.next();
        expect(fv2.has_value());
        expect(fv2->status == FrameStatus::BadSequence);
    };

    "reader flags Truncated when stream cut mid-payload"_test = [] {
        WireStreamWriter       w(0U);
        std::vector<std::byte> out;
        w.writeHeader(out);
        const auto payload = makeBuffer(100UZ);
        w.writeFrame(out, asSpan(payload));
        out.resize(out.size() - 10UZ);

        WireStreamReader r;
        const auto       h = r.begin(out);
        expect(h.has_value());
        const auto fv = r.next();
        expect(fv.has_value());
        expect(fv->status == FrameStatus::Truncated);
    };
};

int main() { return 0; }
