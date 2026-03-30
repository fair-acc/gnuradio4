#include <boost/ut.hpp>

#include <atomic>
#include <complex>
#include <thread>
#include <vector>

#include <gnuradio-4.0/sdr/LoopbackDevice.hpp>

using namespace boost::ut;
using CF32 = std::complex<float>;

const boost::ut::suite<"ChannelModel"> channelModelTests = [] {
    using namespace gr::blocks::sdr::loopback;

    "passthrough copies input to output"_test = [] {
        auto              model = ChannelModel::passthrough();
        std::vector<CF32> in    = {{1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}};
        std::vector<CF32> out(3);
        model.process(in, out);
        expect(eq(out[0], in[0]));
        expect(eq(out[1], in[1]));
        expect(eq(out[2], in[2]));
    };

    "attenuation scales by dB"_test = [] {
        auto              model = ChannelModel::attenuation(-20.f);
        std::vector<CF32> in    = {{1.f, 0.f}, {0.f, 1.f}};
        std::vector<CF32> out(2);
        model.process(in, out);
        expect(approx(out[0].real(), 0.1f, 1e-3f));
        expect(approx(out[1].imag(), 0.1f, 1e-3f));
    };

    "awgn adds noise"_test = [] {
        auto              model = ChannelModel::awgn(-20.f);
        std::vector<CF32> in(1000, CF32{1.f, 0.f});
        std::vector<CF32> out(1000);
        model.process(in, out);
        float sumDiff = 0.f;
        for (std::size_t i = 0UZ; i < in.size(); ++i) {
            sumDiff += std::abs(out[i] - in[i]);
        }
        expect(gt(sumDiff, 0.f));
    };

    "delay shifts samples"_test = [] {
        auto              model = ChannelModel::delay(3);
        std::vector<CF32> in    = {{1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f}, {5.f, 0.f}};
        std::vector<CF32> out(5);
        model.process(in, out);
        expect(eq(out[0], CF32{0.f, 0.f}));
        expect(eq(out[1], CF32{0.f, 0.f}));
        expect(eq(out[2], CF32{0.f, 0.f}));
        expect(eq(out[3], CF32{1.f, 0.f}));
        expect(eq(out[4], CF32{2.f, 0.f}));
    };

    "chain composes models from initializer_list"_test = [] {
        auto              model = ChannelModel::chain({ChannelModel::attenuation(-20.f), ChannelModel::delay(2)});
        std::vector<CF32> in    = {{10.f, 0.f}, {20.f, 0.f}, {30.f, 0.f}, {40.f, 0.f}};
        std::vector<CF32> out(4);
        model.process(in, out);
        expect(eq(out[0], CF32{0.f, 0.f}));
        expect(eq(out[1], CF32{0.f, 0.f}));
        expect(approx(out[2].real(), 1.f, 1e-2f));
        expect(approx(out[3].real(), 2.f, 1e-2f));
    };

    "chain composes models from vector"_test = [] {
        std::vector<ChannelModel> stages;
        stages.push_back(ChannelModel::attenuation(-20.f));
        stages.push_back(ChannelModel::delay(2));
        auto              model = ChannelModel::chain(std::move(stages));
        std::vector<CF32> in    = {{10.f, 0.f}, {20.f, 0.f}, {30.f, 0.f}, {40.f, 0.f}};
        std::vector<CF32> out(4);
        model.process(in, out);
        expect(eq(out[0], CF32{0.f, 0.f}));
        expect(eq(out[1], CF32{0.f, 0.f}));
        expect(approx(out[2].real(), 1.f, 1e-2f));
        expect(approx(out[3].real(), 2.f, 1e-2f));
    };

    "chain with empty stages acts as passthrough"_test = [] {
        auto              model = ChannelModel::chain({});
        std::vector<CF32> in    = {{1.f, 2.f}, {3.f, 4.f}};
        std::vector<CF32> out(2);
        model.process(in, out);
        expect(eq(out[0], in[0]));
        expect(eq(out[1], in[1]));
    };
};

const boost::ut::suite<"LoopbackDevice"> loopbackDeviceTests = [] {
    using namespace gr::blocks::sdr::loopback;

    "direct construction and settings"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(eq(dev.getNumChannels(SOAPY_SDR_RX), 1UZ));
        expect(eq(dev.getDriverKey(), std::string("loopback")));

        dev.setSampleRate(SOAPY_SDR_RX, 0, 2.048e6);
        expect(approx(dev.getSampleRate(SOAPY_SDR_RX, 0), 2.048e6, 1.0));

        dev.setFrequency(SOAPY_SDR_RX, 0, 433.92e6);
        expect(approx(dev.getFrequency(SOAPY_SDR_RX, 0), 433.92e6, 1.0));

        dev.setGain(SOAPY_SDR_RX, 0, 30.0);
        expect(approx(dev.getGain(SOAPY_SDR_RX, 0), 30.0, 0.1));

        auto antennas = dev.listAntennas(SOAPY_SDR_RX, 0);
        expect(!antennas.empty());

        auto rates = dev.listSampleRates(SOAPY_SDR_RX, 0);
        expect(!rates.empty());

        auto formats = dev.getStreamFormats(SOAPY_SDR_RX, 0);
        expect(eq(formats.size(), 3UZ));
    };

    "TX->RX passthrough single channel CF32"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setSampleRate(SOAPY_SDR_TX, 0, 1e6);

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 256;
        std::vector<CF32>     txData(nSamples);
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            txData[i] = {static_cast<float>(i), static_cast<float>(i) * 0.5f};
        }

        int         flags    = 0;
        long long   timeNs   = 0;
        const void* txBufs[] = {txData.data()};
        auto        txRet    = dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);
        expect(eq(txRet, static_cast<int>(nSamples)));

        std::vector<CF32> rxData(nSamples);
        void*             rxBufs[] = {rxData.data()};
        auto              rxRet    = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);
        expect(eq(rxRet, static_cast<int>(nSamples)));

        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            expect(eq(rxData[i], txData[i])) << std::format("mismatch at {}", i);
        }

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "channel model attenuation via writeSetting"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.writeSetting("attenuation_dB", "-20");
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setSampleRate(SOAPY_SDR_TX, 0, 1e6);

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 64;
        std::vector<CF32>     txData(nSamples, CF32{1.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<CF32> rxData(nSamples);
        void*             rxBufs[] = {rxData.data()};
        dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);

        expect(approx(rxData[0].real(), 0.1f, 1e-2f));

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "channel model set programmatically"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.setChannelModel(ChannelModel::attenuation(-6.f)); // ~0.5x

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        std::vector<CF32> txData(32, CF32{2.f, 0.f});
        int               flags    = 0;
        long long         timeNs   = 0;
        const void*       txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, 32, flags, timeNs);

        std::vector<CF32> rxData(32);
        void*             rxBufs[] = {rxData.data()};
        dev.readStream(rxStream, rxBufs, 32, flags, timeNs);

        expect(approx(rxData[0].real(), 1.f, 0.05f)); // 2.0 * 0.5 ~ 1.0

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "per-channel model via writeSetting(direction, channel)"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"num_channels", "2"}});

        // channel 0: -20 dB attenuation, channel 1: passthrough
        dev.writeSetting(SOAPY_SDR_TX, 0, "attenuation_dB", "-20");
        // channel 1 keeps default passthrough

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0, 1});
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {0, 1});
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 64;
        std::vector<CF32>     tx0(nSamples, CF32{1.f, 0.f});
        std::vector<CF32>     tx1(nSamples, CF32{1.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {tx0.data(), tx1.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<CF32> rx0(nSamples), rx1(nSamples);
        void*             rxBufs[] = {rx0.data(), rx1.data()};
        dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);

        // channel 0 attenuated: 1.0 * 0.1 = 0.1
        expect(approx(rx0[0].real(), 0.1f, 1e-2f)) << "channel 0 should be attenuated";
        // channel 1 passthrough: 1.0
        expect(approx(rx1[0].real(), 1.f, 1e-4f)) << "channel 1 should be passthrough";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "per-channel model set programmatically"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"num_channels", "2"}});

        dev.setChannelModel(0UZ, ChannelModel::attenuation(-6.f));  // ~0.5x
        dev.setChannelModel(1UZ, ChannelModel::attenuation(-20.f)); // ~0.1x

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0, 1});
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {0, 1});
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 32;
        std::vector<CF32>     tx0(nSamples, CF32{2.f, 0.f});
        std::vector<CF32>     tx1(nSamples, CF32{2.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {tx0.data(), tx1.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<CF32> rx0(nSamples), rx1(nSamples);
        void*             rxBufs[] = {rx0.data(), rx1.data()};
        dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);

        expect(approx(rx0[0].real(), 1.f, 0.05f)) << "channel 0: 2.0 * 0.5";
        expect(approx(rx1[0].real(), 0.2f, 0.02f)) << "channel 1: 2.0 * 0.1";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "2-channel TX->RX"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"num_channels", "2"}});
        expect(eq(dev.getNumChannels(SOAPY_SDR_RX), 2UZ));

        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setSampleRate(SOAPY_SDR_RX, 1, 1e6);
        dev.setSampleRate(SOAPY_SDR_TX, 0, 1e6);
        dev.setSampleRate(SOAPY_SDR_TX, 1, 1e6);

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0, 1});
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {0, 1});
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 128;
        std::vector<CF32>     tx0(nSamples), tx1(nSamples);
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            tx0[i] = {static_cast<float>(i), 0.f};
            tx1[i] = {0.f, static_cast<float>(i)};
        }
        const void* txBufs[] = {tx0.data(), tx1.data()};
        int         flags    = 0;
        long long   timeNs   = 0;
        auto        ret      = dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);
        expect(eq(ret, static_cast<int>(nSamples)));

        std::vector<CF32> rx0(nSamples), rx1(nSamples);
        void*             rxBufs[] = {rx0.data(), rx1.data()};
        ret                        = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);
        expect(eq(ret, static_cast<int>(nSamples)));

        expect(eq(rx0[10], tx0[10]));
        expect(eq(rx1[10], tx1[10]));

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "TX->RX with CS16 format conversion"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16);
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 32;
        std::vector<int16_t>  txData(nSamples * 2); // I,Q interleaved
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            txData[2 * i]     = static_cast<int16_t>(i * 100);
            txData[2 * i + 1] = static_cast<int16_t>(-(static_cast<int>(i) * 100));
        }

        int         flags    = 0;
        long long   timeNs   = 0;
        const void* txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<int16_t> rxData(nSamples * 2, 0);
        void*                rxBufs[] = {rxData.data()};
        auto                 ret      = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);
        expect(eq(ret, static_cast<int>(nSamples)));
        // round-trip through CF32 loses precision but should be close
        expect(approx(static_cast<float>(rxData[0]), static_cast<float>(txData[0]), 2.f));
        expect(approx(static_cast<float>(rxData[1]), static_cast<float>(txData[1]), 2.f));

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "TX->RX with CU8 format conversion"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CU8);
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CU8);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 32;
        std::vector<uint8_t>  txData(nSamples * 2);
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            txData[2 * i]     = static_cast<uint8_t>(128 + i);
            txData[2 * i + 1] = static_cast<uint8_t>(128 - i);
        }

        int         flags    = 0;
        long long   timeNs   = 0;
        const void* txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<uint8_t> rxData(nSamples * 2, 0);
        void*                rxBufs[] = {rxData.data()};
        auto                 ret      = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);
        expect(eq(ret, static_cast<int>(nSamples)));
        expect(approx(static_cast<float>(rxData[0]), static_cast<float>(txData[0]), 2.f));
        expect(approx(static_cast<float>(rxData[1]), static_cast<float>(txData[1]), 2.f));

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "timing simulation rate-limits readStream"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setSampleRate(SOAPY_SDR_TX, 0, 1e6);
        dev.writeSetting("simulate_timing", "true");

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        // write enough data for two reads
        constexpr std::size_t nSamples = 1000;
        std::vector<CF32>     txData(nSamples * 2, CF32{1.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, nSamples * 2, flags, timeNs);

        // first read should succeed (timing starts from activation)
        std::vector<CF32> rxData(nSamples);
        void*             rxBufs[] = {rxData.data()};
        auto              ret      = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);
        expect(eq(ret, static_cast<int>(nSamples)));

        // second immediate read — at 1 MS/s, 1000 samples = 1 ms, so with 100us timeout it should timeout
        ret = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs, 100);
        expect(eq(ret, SOAPY_SDR_TIMEOUT));

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "readStream returns TIMEOUT when no TX data"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);

        std::vector<CF32> rxData(64);
        void*             rxBufs[] = {rxData.data()};
        int               flags    = 0;
        long long         timeNs   = 0;
        auto              ret      = dev.readStream(rxStream, rxBufs, 64, flags, timeNs, 1000);
        expect(eq(ret, SOAPY_SDR_TIMEOUT));

        dev.deactivateStream(rxStream);
    };

    "buffer drain on deactivate clears stale data"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        // write some data but don't read it
        constexpr std::size_t nSamples = 128;
        std::vector<CF32>     txData(nSamples, CF32{42.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        // deactivate RX (drains buffers), then reactivate
        dev.deactivateStream(rxStream);
        dev.activateStream(rxStream);

        // should get TIMEOUT — stale data was drained
        std::vector<CF32> rxData(nSamples);
        void*             rxBufs[] = {rxData.data()};
        auto              ret      = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs, 1000);
        expect(eq(ret, SOAPY_SDR_TIMEOUT)) << "stale data should have been drained";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "writeStream on inactive TX stream returns STREAM_ERROR"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        // do NOT activate
        (void)txStream;

        std::vector<CF32> txData(64, CF32{1.f, 0.f});
        int               flags    = 0;
        long long         timeNs   = 0;
        const void*       txBufs[] = {txData.data()};
        auto              ret      = dev.writeStream(txStream, txBufs, 64, flags, timeNs);
        expect(eq(ret, SOAPY_SDR_STREAM_ERROR));
    };

    "readStream on inactive RX stream returns STREAM_ERROR"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        // do NOT activate

        std::vector<CF32> rxData(64);
        void*             rxBufs[] = {rxData.data()};
        int               flags    = 0;
        long long         timeNs   = 0;
        auto              ret      = dev.readStream(rxStream, rxBufs, 64, flags, timeNs);
        expect(eq(ret, SOAPY_SDR_STREAM_ERROR));
    };

    "custom buffer_size kwarg"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"buffer_size", "1024"}});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        expect(eq(dev.getStreamMTU(rxStream), 1024UZ));
    };

    "concurrent TX and RX threads"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setSampleRate(SOAPY_SDR_TX, 0, 1e6);

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t kTotalSamples = 4096UZ;
        constexpr std::size_t kChunkSize    = 256UZ;

        std::vector<CF32> txData(kTotalSamples);
        for (std::size_t i = 0UZ; i < kTotalSamples; ++i) {
            txData[i] = {static_cast<float>(i), static_cast<float>(i) * 0.5f};
        }

        std::vector<CF32>        rxData(kTotalSamples);
        std::atomic<std::size_t> rxTotal{0UZ};

        auto txThread = std::jthread([&] {
            std::size_t written = 0UZ;
            while (written < kTotalSamples) {
                auto        nToWrite = std::min(kChunkSize, kTotalSamples - written);
                const void* bufs[]   = {txData.data() + written};
                int         flags    = 0;
                long long   timeNs   = 0;
                auto        ret      = dev.writeStream(txStream, bufs, nToWrite, flags, timeNs);
                if (ret > 0) {
                    written += static_cast<std::size_t>(ret);
                }
            }
        });

        auto rxThread = std::jthread([&] {
            std::size_t received = 0UZ;
            while (received < kTotalSamples) {
                auto      nToRead = std::min(kChunkSize, kTotalSamples - received);
                void*     bufs[]  = {rxData.data() + received};
                int       flags   = 0;
                long long timeNs  = 0;
                auto      ret     = dev.readStream(rxStream, bufs, nToRead, flags, timeNs, 10000);
                if (ret > 0) {
                    received += static_cast<std::size_t>(ret);
                }
            }
            rxTotal.store(received, std::memory_order_release);
        });

        txThread.join();
        rxThread.join();

        expect(eq(rxTotal.load(), kTotalSamples));
        for (std::size_t i = 0UZ; i < kTotalSamples; ++i) {
            expect(eq(rxData[i], txData[i])) << std::format("mismatch at {}", i);
        }

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "writeStream returns TIMEOUT when RX buffer is full"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"buffer_size", "512"}});
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        dev.activateStream(txStream);
        dev.activateStream(rxStream);

        std::vector<CF32> txData(64, CF32{1.f, 0.f});
        int               flags    = 0;
        long long         timeNs   = 0;
        const void*       txBufs[] = {txData.data()};

        bool gotTimeout = false;
        for (int attempt = 0; attempt < 100; ++attempt) {
            auto ret = dev.writeStream(txStream, txBufs, 64, flags, timeNs);
            if (ret == SOAPY_SDR_TIMEOUT) {
                gotTimeout = true;
                break;
            }
            expect(gt(ret, 0));
        }
        expect(gotTimeout) << "should eventually get TIMEOUT from backpressure";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "CS16 format conversion handles boundary values"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16);
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        std::vector<int16_t> txData = {
            32767,
            -32768, // sample 0: max I, min Q
            0,
            0, // sample 1: zero
            16384,
            -16384, // sample 2: mid-range
        };
        int         flags    = 0;
        long long   timeNs   = 0;
        const void* txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, 3, flags, timeNs);

        std::vector<int16_t> rxData(6, 0);
        void*                rxBufs[] = {rxData.data()};
        auto                 ret      = dev.readStream(rxStream, rxBufs, 3, flags, timeNs);
        expect(eq(ret, 3));

        expect(approx(static_cast<float>(rxData[0]), 32767.f, 2.f)) << "I=INT16_MAX";
        expect(approx(static_cast<float>(rxData[1]), -32768.f, 2.f)) << "Q=INT16_MIN";
        expect(eq(rxData[2], static_cast<int16_t>(0))) << "I=0";
        expect(eq(rxData[3], static_cast<int16_t>(0))) << "Q=0";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "CU8 format conversion handles boundary values"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CU8);
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CU8);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        std::vector<uint8_t> txData = {
            255,
            0, // sample 0: max I, min Q
            128,
            128, // sample 1: centre (zero in CF32)
            0,
            255, // sample 2: min I, max Q
        };
        int         flags    = 0;
        long long   timeNs   = 0;
        const void* txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, 3, flags, timeNs);

        std::vector<uint8_t> rxData(6, 0);
        void*                rxBufs[] = {rxData.data()};
        auto                 ret      = dev.readStream(rxStream, rxBufs, 3, flags, timeNs);
        expect(eq(ret, 3));

        expect(approx(static_cast<float>(rxData[0]), 255.f, 2.f)) << "I=max";
        expect(approx(static_cast<float>(rxData[1]), 0.f, 2.f)) << "Q=min";
        expect(approx(static_cast<float>(rxData[2]), 128.f, 2.f)) << "I=centre";
        expect(approx(static_cast<float>(rxData[3]), 128.f, 2.f)) << "Q=centre";
        expect(approx(static_cast<float>(rxData[4]), 0.f, 2.f)) << "I=min";
        expect(approx(static_cast<float>(rxData[5]), 255.f, 2.f)) << "Q=max";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };
};

const boost::ut::suite<"RxOnlyDevice"> rxOnlyTests = [] {
    using namespace gr::blocks::sdr::loopback;

    "generates tone without TX"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"device_mode", "rx_only"}});
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setFrequency(SOAPY_SDR_RX, 0, 100e3); // 100 kHz tone

        expect(eq(dev.readSetting("device_mode"), std::string("rx_only")));

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);

        constexpr std::size_t nSamples = 1024;
        std::vector<CF32>     rxData(nSamples);
        void*                 rxBufs[] = {rxData.data()};
        int                   flags    = 0;
        long long             timeNs   = 0;
        auto                  ret      = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);
        expect(eq(ret, static_cast<int>(nSamples)));

        float maxAbs = 0.f;
        for (const auto& s : rxData) {
            maxAbs = std::max(maxAbs, std::abs(s));
        }
        expect(approx(maxAbs, 1.f, 0.01f)) << "should produce unit-amplitude tone";

        dev.deactivateStream(rxStream);
    };

    "applies channel model to generated tone"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"device_mode", "rx_only"}});
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setFrequency(SOAPY_SDR_RX, 0, 100e3);
        dev.setChannelModel(ChannelModel::attenuation(-20.f)); // 0.1x

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);

        constexpr std::size_t nSamples = 256;
        std::vector<CF32>     rxData(nSamples);
        void*                 rxBufs[] = {rxData.data()};
        int                   flags    = 0;
        long long             timeNs   = 0;
        dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);

        float maxAbs = 0.f;
        for (const auto& s : rxData) {
            maxAbs = std::max(maxAbs, std::abs(s));
        }
        expect(approx(maxAbs, 0.1f, 0.01f)) << "channel model should attenuate tone";

        dev.deactivateStream(rxStream);
    };

    "device_mode setting round-trip"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(eq(dev.readSetting("device_mode"), std::string("loopback")));
        dev.writeSetting("device_mode", "rx_only");
        expect(eq(dev.readSetting("device_mode"), std::string("rx_only")));
        dev.writeSetting("device_mode", "tx_only");
        expect(eq(dev.readSetting("device_mode"), std::string("tx_only")));
        dev.writeSetting("device_mode", "loopback");
        expect(eq(dev.readSetting("device_mode"), std::string("loopback")));
    };
};

const boost::ut::suite<"TxOnlyDevice"> txOnlyTests = [] {
    using namespace gr::blocks::sdr::loopback;

    "accepts and discards TX data"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"device_mode", "tx_only"}});

        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        dev.activateStream(txStream);
        dev.activateStream(rxStream);

        constexpr std::size_t nSamples = 256;
        std::vector<CF32>     txData(nSamples, CF32{1.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {txData.data()};
        auto                  txRet    = dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);
        expect(eq(txRet, static_cast<int>(nSamples))) << "should accept all samples";

        std::vector<CF32> rxData(nSamples);
        void*             rxBufs[] = {rxData.data()};
        auto              rxRet    = dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs, 1000);
        expect(eq(rxRet, SOAPY_SDR_TIMEOUT)) << "should not route data to RX";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };
};

const boost::ut::suite<"DeviceRegistry"> registryTests = [] {
    using namespace gr::blocks::sdr::loopback;

    "findOrCreate returns same instance for same ID"_test = [] {
        auto dev1 = DeviceRegistry::findOrCreate(100UZ, {{"driver", "loopback#100"}});
        auto dev2 = DeviceRegistry::findOrCreate(100UZ, {{"driver", "loopback#100"}});
        expect(eq(dev1.get(), dev2.get())) << "same ID should return same instance";
        expect(eq(dev1->instanceId(), 100UZ));
    };

    "findOrCreate returns different instances for different IDs"_test = [] {
        auto dev1 = DeviceRegistry::findOrCreate(200UZ, {{"driver", "loopback#200"}});
        auto dev2 = DeviceRegistry::findOrCreate(201UZ, {{"driver", "loopback#201"}});
        expect(neq(dev1.get(), dev2.get())) << "different IDs should return different instances";
    };

    "instance is released when all shared_ptrs expire"_test = [] {
        {
            auto dev = DeviceRegistry::findOrCreate(300UZ, {{"driver", "loopback#300"}});
            expect(eq(dev->instanceId(), 300UZ));
        }
        // after dev goes out of scope, next findOrCreate should create a new instance
        auto dev2 = DeviceRegistry::findOrCreate(300UZ, {{"driver", "loopback#300"}});
        expect(eq(dev2->instanceId(), 300UZ)); // still works, new instance
    };

    "shared instance supports TX->RX loopback"_test = [] {
        auto dev = DeviceRegistry::findOrCreate(400UZ, {{"driver", "loopback#400"}});

        auto* txStream = dev->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        auto* rxStream = dev->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        dev->activateStream(txStream);
        dev->activateStream(rxStream);

        constexpr std::size_t nSamples = 64;
        std::vector<CF32>     txData(nSamples, CF32{7.f, 3.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {txData.data()};
        dev->writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<CF32> rxData(nSamples);
        void*             rxBufs[] = {rxData.data()};
        auto              ret      = dev->readStream(rxStream, rxBufs, nSamples, flags, timeNs);
        expect(eq(ret, static_cast<int>(nSamples)));
        expect(eq(rxData[0], CF32{7.f, 3.f}));

        dev->deactivateStream(rxStream);
        dev->deactivateStream(txStream);
    };

    "parseInstanceId extracts ID from driver kwarg"_test = [] {
        expect(eq(DeviceRegistry::parseInstanceId({{"driver", "loopback"}}), 0UZ));
        expect(eq(DeviceRegistry::parseInstanceId({{"driver", "loopback#0"}}), 0UZ));
        expect(eq(DeviceRegistry::parseInstanceId({{"driver", "loopback#5"}}), 5UZ));
        expect(eq(DeviceRegistry::parseInstanceId({{"driver", "loopback#42"}}), 42UZ));
        expect(eq(DeviceRegistry::parseInstanceId({}), 0UZ));
    };

    "isLoopbackDriver matches loopback variants"_test = [] {
        expect(DeviceRegistry::isLoopbackDriver({{"driver", "loopback"}}));
        expect(DeviceRegistry::isLoopbackDriver({{"driver", "loopback#0"}}));
        expect(DeviceRegistry::isLoopbackDriver({{"driver", "loopback#99"}}));
        expect(!DeviceRegistry::isLoopbackDriver({{"driver", "rtlsdr"}}));
        expect(!DeviceRegistry::isLoopbackDriver({}));
    };
};

const boost::ut::suite<"SoapySDR API completeness"> apiTests = [] {
    using namespace gr::blocks::sdr::loopback;

    "identification API"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(eq(dev.getDriverKey(), std::string("loopback")));
        expect(eq(dev.getHardwareKey(), std::string("loopback")));
        auto info = dev.getHardwareInfo();
        expect(eq(info.at("driver"), std::string("loopback")));
        expect(eq(info.at("version"), std::string("1.0")));
    };

    "frontend mapping"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(eq(dev.getFrontendMapping(SOAPY_SDR_RX), std::string("")));
        dev.setFrontendMapping(SOAPY_SDR_RX, "0:0");
        expect(eq(dev.getFrontendMapping(SOAPY_SDR_RX), std::string("0:0")));
        expect(eq(dev.getFrontendMapping(SOAPY_SDR_TX), std::string("")));
        dev.setFrontendMapping(SOAPY_SDR_TX, "1:1");
        expect(eq(dev.getFrontendMapping(SOAPY_SDR_TX), std::string("1:1")));
    };

    "channel info"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"num_channels", "2"}});
        expect(eq(dev.getNumChannels(SOAPY_SDR_RX), 2UZ));
        expect(eq(dev.getNumChannels(SOAPY_SDR_TX), 2UZ));
        expect(dev.getFullDuplex(SOAPY_SDR_RX, 0));
        expect(dev.getFullDuplex(SOAPY_SDR_TX, 1));
    };

    "gain range"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           range = dev.getGainRange(SOAPY_SDR_RX, 0);
        expect(approx(range.minimum(), 0.0, 0.1));
        expect(approx(range.maximum(), 60.0, 0.1));
        auto namedRange = dev.getGainRange(SOAPY_SDR_RX, 0, "TUNER");
        expect(approx(namedRange.maximum(), 60.0, 0.1));
    };

    "frequency named component delegates to overall"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.setFrequency(SOAPY_SDR_RX, 0, "RF", 100e6);
        expect(approx(dev.getFrequency(SOAPY_SDR_RX, 0), 100e6, 1.0));
        expect(approx(dev.getFrequency(SOAPY_SDR_RX, 0, "RF"), 100e6, 1.0));
        auto range = dev.getFrequencyRange(SOAPY_SDR_RX, 0, "RF");
        expect(!range.empty());
    };

    "sample rate range"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           range = dev.getSampleRateRange(SOAPY_SDR_RX, 0);
        expect(!range.empty());
        expect(approx(range[0].minimum(), 250e3, 1.0));
        expect(approx(range[0].maximum(), 20e6, 1.0));
    };

    "bandwidth range"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           range = dev.getBandwidthRange(SOAPY_SDR_RX, 0);
        expect(!range.empty());
        expect(approx(range[0].minimum(), 200e3, 1.0));
    };

    "native stream format"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        double         fullScale = 0.0;
        auto           format    = dev.getNativeStreamFormat(SOAPY_SDR_RX, 0, fullScale);
        expect(eq(format, std::string(SOAPY_SDR_CF32)));
        expect(approx(fullScale, 1.0, 1e-6));
    };

    "stream args info returns empty"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(dev.getStreamArgsInfo(SOAPY_SDR_RX, 0).empty());
    };

    "DC offset mode store and recall"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(dev.hasDCOffsetMode(SOAPY_SDR_RX, 0));
        expect(!dev.getDCOffsetMode(SOAPY_SDR_RX, 0));
        dev.setDCOffsetMode(SOAPY_SDR_RX, 0, true);
        expect(dev.getDCOffsetMode(SOAPY_SDR_RX, 0));
    };

    "DC offset value store and recall"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(dev.hasDCOffset(SOAPY_SDR_RX, 0));
        dev.setDCOffset(SOAPY_SDR_RX, 0, {0.01, -0.02});
        auto offset = dev.getDCOffset(SOAPY_SDR_RX, 0);
        expect(approx(offset.real(), 0.01, 1e-9));
        expect(approx(offset.imag(), -0.02, 1e-9));
    };

    "IQ balance store and recall"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(dev.hasIQBalance(SOAPY_SDR_RX, 0));
        dev.setIQBalance(SOAPY_SDR_RX, 0, {0.98, 0.01});
        auto balance = dev.getIQBalance(SOAPY_SDR_RX, 0);
        expect(approx(balance.real(), 0.98, 1e-9));
        expect(approx(balance.imag(), 0.01, 1e-9));
        expect(!dev.hasIQBalanceMode(SOAPY_SDR_RX, 0));
    };

    "frequency correction (ppm)"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(dev.hasFrequencyCorrection(SOAPY_SDR_RX, 0));
        expect(approx(dev.getFrequencyCorrection(SOAPY_SDR_RX, 0), 0.0, 1e-9));
        dev.setFrequencyCorrection(SOAPY_SDR_RX, 0, 1.5);
        expect(approx(dev.getFrequencyCorrection(SOAPY_SDR_RX, 0), 1.5, 1e-6));
    };

    "time and clock sources"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(!dev.hasHardwareTime());
        expect(eq(dev.getHardwareTime(), 0LL));
        auto timeSources = dev.listTimeSources();
        expect(!timeSources.empty());
        expect(eq(dev.getTimeSource(), std::string("none")));
        auto clockSources = dev.listClockSources();
        expect(eq(clockSources.size(), 2UZ));
        expect(eq(dev.getClockSource(), std::string("internal")));
        dev.setClockSource("external");
        expect(eq(dev.getClockSource(), std::string("external")));
    };

    "master and reference clock rates"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.setMasterClockRate(40e6);
        expect(approx(dev.getMasterClockRate(), 40e6, 1.0));
        expect(!dev.getMasterClockRates().empty());
        dev.setReferenceClockRate(10e6);
        expect(approx(dev.getReferenceClockRate(), 10e6, 1.0));
        expect(!dev.getReferenceClockRates().empty());
    };

    "stream status returns timeout"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        size_t    chanMask = 0;
        int       flags    = 0;
        long long timeNs   = 0;
        auto      ret      = dev.readStreamStatus(rxStream, chanMask, flags, timeNs, 1);
        expect(eq(ret, SOAPY_SDR_TIMEOUT));
        dev.deactivateStream(rxStream);
    };

    "device sensors"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           sensors = dev.listSensors();
        expect(eq(sensors.size(), 2UZ));
        auto tempInfo = dev.getSensorInfo("temperature");
        expect(eq(tempInfo.key, std::string("temperature")));
        expect(eq(dev.readSensor("temperature"), std::string("25.0")));
        expect(eq(dev.readSensor("lo_locked"), std::string("true")));
    };

    "per-channel sensors"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           sensors = dev.listSensors(SOAPY_SDR_RX, 0);
        expect(eq(sensors.size(), 1UZ));
        auto rssiInfo = dev.getSensorInfo(SOAPY_SDR_RX, 0, "rssi");
        expect(eq(rssiInfo.key, std::string("rssi")));
        expect(eq(dev.readSensor(SOAPY_SDR_RX, 0, "rssi"), std::string("-60.0")));
    };

    "register read/write"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           ifaces = dev.listRegisterInterfaces();
        expect(!ifaces.empty());
        dev.writeRegister("loopback_regs", 0x10, 0xDEAD);
        expect(eq(dev.readRegister("loopback_regs", 0x10), 0xDEADu));
        expect(eq(dev.readRegister("loopback_regs", 0x20), 0u));
        dev.writeRegisters("loopback_regs", 0x00, {0x11, 0x22, 0x33});
        auto regs = dev.readRegisters("loopback_regs", 0x00, 3);
        expect(eq(regs.size(), 3UZ));
        expect(eq(regs[0], 0x11u));
        expect(eq(regs[2], 0x33u));
    };

    "GPIO read/write with masking"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           banks = dev.listGPIOBanks();
        expect(!banks.empty());
        dev.writeGPIO("MAIN", 0xFF);
        expect(eq(dev.readGPIO("MAIN"), 0xFFu));
        dev.writeGPIO("MAIN", 0x00, 0x0F); // clear lower nibble
        expect(eq(dev.readGPIO("MAIN"), 0xF0u));
        dev.writeGPIODir("MAIN", 0xFF);
        expect(eq(dev.readGPIODir("MAIN"), 0xFFu));
        dev.writeGPIODir("MAIN", 0x00, 0x0F);
        expect(eq(dev.readGPIODir("MAIN"), 0xF0u));
    };

    "I2C, SPI, and UART stubs"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.writeI2C(0x50, "hello");
        auto i2cData = dev.readI2C(0x50, 4);
        expect(eq(i2cData.size(), 4UZ));
        auto spiResult = dev.transactSPI(0, 0xAB, 8);
        expect(eq(spiResult, 0xABu));
        expect(dev.listUARTs().empty());
    };

    "device-wide settings info"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           infos = dev.getSettingInfo();
        expect(ge(infos.size(), 2UZ));
    };

    "per-channel settings info"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto           infos = dev.getSettingInfo(SOAPY_SDR_RX, 0);
        expect(ge(infos.size(), 2UZ));
    };

    "readSetting round-trip"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(eq(dev.readSetting("simulate_timing"), std::string("false")));
        dev.writeSetting("simulate_timing", "true");
        expect(eq(dev.readSetting("simulate_timing"), std::string("true")));
        dev.writeSetting("simulate_timing", "false");
        expect(eq(dev.readSetting("simulate_timing"), std::string("false")));
    };

    "writeSetting noise_floor_dBFS applies AWGN model"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.writeSetting("noise_floor_dBFS", "-20");
        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 256;
        std::vector<CF32>     txData(nSamples, CF32{1.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<CF32> rxData(nSamples);
        void*             rxBufs[] = {rxData.data()};
        dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);

        float sumDiff = 0.f;
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            sumDiff += std::abs(rxData[i] - txData[i]);
        }
        expect(gt(sumDiff, 0.f)) << "AWGN should add noise";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "writeSetting delay_samples applies delay model"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.writeSetting("delay_samples", "3");
        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        std::vector<CF32> txData   = {{1.f, 0.f}, {2.f, 0.f}, {3.f, 0.f}, {4.f, 0.f}, {5.f, 0.f}};
        int               flags    = 0;
        long long         timeNs   = 0;
        const void*       txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, 5, flags, timeNs);

        std::vector<CF32> rxData(5);
        void*             rxBufs[] = {rxData.data()};
        dev.readStream(rxStream, rxBufs, 5, flags, timeNs);

        expect(eq(rxData[0], CF32{0.f, 0.f})) << "first 3 should be zero (delay)";
        expect(eq(rxData[3], CF32{1.f, 0.f})) << "delayed data starts at index 3";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "per-channel writeSetting noise_floor_dBFS"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{{"num_channels", "2"}});
        dev.writeSetting(SOAPY_SDR_TX, 0, "noise_floor_dBFS", "-20");

        auto* rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0, 1});
        auto* txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {0, 1});
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nSamples = 128;
        std::vector<CF32>     tx0(nSamples, CF32{1.f, 0.f});
        std::vector<CF32>     tx1(nSamples, CF32{1.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {tx0.data(), tx1.data()};
        dev.writeStream(txStream, txBufs, nSamples, flags, timeNs);

        std::vector<CF32> rx0(nSamples), rx1(nSamples);
        void*             rxBufs[] = {rx0.data(), rx1.data()};
        dev.readStream(rxStream, rxBufs, nSamples, flags, timeNs);

        float diff0 = 0.f, diff1 = 0.f;
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            diff0 += std::abs(rx0[i] - tx0[i]);
            diff1 += std::abs(rx1[i] - tx1[i]);
        }
        expect(gt(diff0, 0.f)) << "channel 0 should have noise";
        expect(approx(diff1, 0.f, 1e-6f)) << "channel 1 should be passthrough";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "out-of-range channel in setChannelModel is safe"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        dev.setChannelModel(99UZ, ChannelModel::passthrough());      // should not crash
        dev.writeSetting(SOAPY_SDR_TX, 99, "attenuation_dB", "-20"); // should not crash
    };

    "single-sample TX->RX round-trip"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        CF32        txSample = {42.f, -7.f};
        int         flags    = 0;
        long long   timeNs   = 0;
        const void* txBufs[] = {&txSample};
        auto        txRet    = dev.writeStream(txStream, txBufs, 1, flags, timeNs);
        expect(eq(txRet, 1));

        CF32  rxSample = {0.f, 0.f};
        void* rxBufs[] = {&rxSample};
        auto  rxRet    = dev.readStream(rxStream, rxBufs, 1, flags, timeNs);
        expect(eq(rxRet, 1));
        expect(eq(rxSample, txSample));

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "partial read returns only available samples"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        auto*          rxStream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        auto*          txStream = dev.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32);
        dev.activateStream(rxStream);
        dev.activateStream(txStream);

        constexpr std::size_t nTx = 10;
        std::vector<CF32>     txData(nTx, CF32{1.f, 0.f});
        int                   flags    = 0;
        long long             timeNs   = 0;
        const void*           txBufs[] = {txData.data()};
        dev.writeStream(txStream, txBufs, nTx, flags, timeNs);

        std::vector<CF32> rxData(100, CF32{0.f, 0.f});
        void*             rxBufs[] = {rxData.data()};
        auto              ret      = dev.readStream(rxStream, rxBufs, 100, flags, timeNs);
        expect(eq(ret, static_cast<int>(nTx))) << "should return only available samples";

        dev.deactivateStream(rxStream);
        dev.deactivateStream(txStream);
    };

    "getNativeDeviceHandle returns nullptr"_test = [] {
        LoopbackDevice dev(SoapySDR::Kwargs{});
        expect(eq(dev.getNativeDeviceHandle(), static_cast<void*>(nullptr)));
    };
};

int main() { /* not needed for UT */ }
