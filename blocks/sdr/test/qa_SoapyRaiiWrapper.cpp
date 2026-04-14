#include <boost/ut.hpp>

#include <complex>
#include <vector>

#include <gnuradio-4.0/sdr/SoapyRaiiWrapper.hpp>

using namespace boost::ut;
using namespace gr::blocks::sdr;
using CF32 = std::complex<float>;

const boost::ut::suite<"SoapyRaiiWrapper module loading"> moduleTests = [] {
    "loopback module is discoverable"_test = [] {
        auto modules = soapy::getSoapySDRModules();
        bool found   = false;
        for (const auto& m : modules) {
            if (m.find("loopback") != std::string::npos) {
                found = true;
                break;
            }
        }
        const char* pluginPath     = std::getenv("SOAPY_SDR_PLUGIN_PATH");
        std::string pluginPathText = (pluginPath != nullptr && *pluginPath != '\0') ? pluginPath : "<unset>";
        expect(found) << ("gr-sdr-loopback.so should be in SOAPY_SDR_PLUGIN_PATH (current: " + pluginPathText + ")");
    };

    "enumerate finds loopback driver"_test = [] {
        auto results = soapy::Device::enumerate({{"driver", "loopback"}});
        expect(!results.empty()) << "loopback driver should be enumerable";
        if (!results.empty()) {
            expect(results[0].contains("driver"));
            if (results[0].contains("driver")) {
                expect(results[0].at("driver").starts_with("loopback"));
            }
        }
    };

    "enumerate with string args"_test = [] {
        auto results = soapy::Device::enumerate("driver=loopback");
        expect(!results.empty());
    };
};

const boost::ut::suite<"SoapyRaiiWrapper Device"> deviceTests = [] {
    "Device::make returns valid device"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value()) << "make should succeed for loopback";
        if (result.has_value()) {
            expect(neq(result->get(), static_cast<SoapySDRDevice*>(nullptr)));
        }
    };

    "Device::make with invalid driver fails"_test = [] {
        auto result = soapy::Device::make({{"driver", "nonexistent_driver_xyz"}});
        expect(!result.has_value());
    };

    "Device RAII cleanup on reset"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        result->reset();
        expect(eq(result->get(), static_cast<SoapySDRDevice*>(nullptr)));
    };

    "Device move semantics"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto dev2 = std::move(*result);
        expect(neq(dev2.get(), static_cast<SoapySDRDevice*>(nullptr)));
        expect(eq(result->get(), static_cast<SoapySDRDevice*>(nullptr)));
    };
};

const boost::ut::suite<"SoapyRaiiWrapper channel + frontend"> channelTests = [] {
    "getNumChannels and getFullDuplex"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        expect(eq(dev.getNumChannels(SOAPY_SDR_RX), 1UZ));
        expect(eq(dev.getNumChannels(SOAPY_SDR_TX), 1UZ));
        expect(dev.getFullDuplex(SOAPY_SDR_RX, 0));
    };

    "frontend mapping round-trip"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        dev.setFrontendMapping(SOAPY_SDR_RX, "0:0");
        expect(eq(dev.getFrontendMapping(SOAPY_SDR_RX), std::string("0:0")));
    };

    "channel info"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto info = result->getChannelInfo(SOAPY_SDR_RX, 0);
        // loopback returns empty channel info, just verify no crash
        (void)info;
    };
};

const boost::ut::suite<"SoapyRaiiWrapper antenna + gain"> antennaGainTests = [] {
    "antenna list, set, get"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev      = *result;
        auto  antennas = dev.listAvailableAntennas(SOAPY_SDR_RX, 0);
        expect(!antennas.empty());
        auto r = dev.setAntenna(SOAPY_SDR_RX, 0, antennas[0]);
        expect(r.has_value());
        expect(eq(dev.getAntenna(SOAPY_SDR_RX, 0), antennas[0]));
    };

    "gain elements and AGC"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev   = *result;
        auto  gains = dev.listAvailableGainElements(SOAPY_SDR_RX, 0);
        expect(!gains.empty());
        expect(dev.hasAutomaticGainControl(SOAPY_SDR_RX, 0));
        auto r = dev.setAutomaticGainControl(SOAPY_SDR_RX, 0, false);
        expect(r.has_value());
        expect(!dev.isAutomaticGainControl(SOAPY_SDR_RX, 0));
    };

    "gain set and get (overall + element)"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        auto  r   = dev.setGain(SOAPY_SDR_RX, 0, 30.0);
        expect(r.has_value());
        expect(approx(dev.getGain(SOAPY_SDR_RX, 0), 30.0, 0.1));
        r = dev.setGain(SOAPY_SDR_RX, 0, 20.0, "TUNER");
        expect(r.has_value());
        expect(approx(dev.getGain(SOAPY_SDR_RX, 0, "TUNER"), 20.0, 0.1));
    };
};

const boost::ut::suite<"SoapyRaiiWrapper frequency + sample rate + bandwidth"> tuningTests = [] {
    "frequency set, get, range"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        auto  r   = dev.setCenterFrequency(SOAPY_SDR_RX, 0, 433.92e6);
        expect(r.has_value());
        expect(approx(dev.getCenterFrequency(SOAPY_SDR_RX, 0), 433.92e6, 1.0));
        auto range = dev.getOverallFrequencyRange(SOAPY_SDR_RX, 0);
        expect(!range.empty());
    };

    "sample rate set, get, list"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev   = *result;
        auto  rates = dev.listSampleRates(SOAPY_SDR_RX, 0);
        expect(!rates.empty());
        auto r = dev.setSampleRate(SOAPY_SDR_RX, 0, 2.048e6);
        expect(r.has_value());
        expect(approx(dev.getSampleRate(SOAPY_SDR_RX, 0), 2.048e6, 1.0));
    };

    "bandwidth set, get, list"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        auto  bws = dev.listAvailableBandwidths(SOAPY_SDR_RX, 0);
        expect(!bws.empty());
        auto r = dev.setBandwidth(SOAPY_SDR_RX, 0, 1e6);
        expect(r.has_value());
        expect(approx(dev.getBandwidth(SOAPY_SDR_RX, 0), 1e6, 1.0));
    };
};

const boost::ut::suite<"SoapyRaiiWrapper frontend corrections"> correctionTests = [] {
    "DC offset mode and value"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        expect(dev.hasDCOffsetMode(SOAPY_SDR_RX, 0));
        auto r = dev.setDCOffsetMode(SOAPY_SDR_RX, 0, true);
        expect(r.has_value());
        expect(dev.getDCOffsetMode(SOAPY_SDR_RX, 0));
        expect(dev.hasDCOffset(SOAPY_SDR_RX, 0));
        r = dev.setDCOffset(SOAPY_SDR_RX, 0, 0.01, -0.02);
        expect(r.has_value());
        auto [oi, oq] = dev.getDCOffset(SOAPY_SDR_RX, 0);
        expect(approx(oi, 0.01, 1e-6));
        expect(approx(oq, -0.02, 1e-6));
    };

    "IQ balance"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        expect(dev.hasIQBalance(SOAPY_SDR_RX, 0));
        auto r = dev.setIQBalance(SOAPY_SDR_RX, 0, 0.98, 0.01);
        expect(r.has_value());
        auto [bi, bq] = dev.getIQBalance(SOAPY_SDR_RX, 0);
        expect(approx(bi, 0.98, 1e-6));
        expect(approx(bq, 0.01, 1e-6));
    };

    "frequency correction (ppm)"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        expect(dev.hasFrequencyCorrection(SOAPY_SDR_RX, 0));
        auto r = dev.setFrequencyCorrection(SOAPY_SDR_RX, 0, 1.5);
        expect(r.has_value());
        expect(approx(dev.getFrequencyCorrection(SOAPY_SDR_RX, 0), 1.5, 1e-6));
    };
};

const boost::ut::suite<"SoapyRaiiWrapper clock + time"> clockTests = [] {
    "master clock rate"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        auto  r   = dev.setMasterClockRate(40e6);
        expect(r.has_value());
        expect(approx(dev.getMasterClockRate(), 40e6, 1.0));
        auto rates = dev.getMasterClockRates();
        expect(!rates.empty());
    };

    "reference clock rate"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev = *result;
        auto  r   = dev.setReferenceClockRate(10e6);
        expect(r.has_value());
        expect(approx(dev.getReferenceClockRate(), 10e6, 1.0));
    };

    "clock sources"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev     = *result;
        auto  sources = dev.listClockSources();
        expect(!sources.empty());
        auto r = dev.setClockSource("external");
        expect(r.has_value());
        expect(eq(dev.getClockSource(), std::string("external")));
    };

    "time sources"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev     = *result;
        auto  sources = dev.listAvailableTimeSources();
        expect(!sources.empty());
        expect(eq(dev.getTimeSource(), std::string("none")));
        expect(eq(dev.getHardwareTime(), 0ULL));
    };
};

const boost::ut::suite<"SoapyRaiiWrapper sensors"> sensorTests = [] {
    "device-level sensors"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev     = *result;
        auto  sensors = dev.listSensors();
        expect(eq(sensors.size(), 2UZ));
        auto info = dev.getSensorInfo("temperature");
        expect(eq(info.key, std::string("temperature")));
        expect(eq(dev.readSensor("temperature"), std::string("25.0")));
        expect(eq(dev.readSensor("lo_locked"), std::string("true")));
    };

    "per-channel sensors"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev     = *result;
        auto  sensors = dev.listChannelSensors(SOAPY_SDR_RX, 0);
        expect(eq(sensors.size(), 1UZ));
        auto info = dev.getChannelSensorInfo(SOAPY_SDR_RX, 0, "rssi");
        expect(eq(info.key, std::string("rssi")));
        expect(eq(dev.readChannelSensor(SOAPY_SDR_RX, 0, "rssi"), std::string("-60.0")));
    };
};

const boost::ut::suite<"SoapyRaiiWrapper GPIO + register"> gpioRegTests = [] {
    "GPIO read/write with masking"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev   = *result;
        auto  banks = dev.listGPIOBanks();
        expect(!banks.empty());

        auto r = dev.writeGPIO("MAIN", 0xFF);
        expect(r.has_value());
        expect(eq(dev.readGPIO("MAIN"), 0xFFu));

        r = dev.writeGPIO("MAIN", 0x00, 0x0F); // clear lower nibble
        expect(r.has_value());
        expect(eq(dev.readGPIO("MAIN"), 0xF0u));

        r = dev.writeGPIODir("MAIN", 0xFF);
        expect(r.has_value());
        expect(eq(dev.readGPIODir("MAIN"), 0xFFu));

        r = dev.writeGPIODir("MAIN", 0x00, 0x0F);
        expect(r.has_value());
        expect(eq(dev.readGPIODir("MAIN"), 0xF0u));
    };

    "register interfaces and read/write"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev    = *result;
        auto  ifaces = dev.listRegisterInterfaces();
        expect(!ifaces.empty());

        auto r = dev.writeRegister("loopback_regs", 0x10, 0xDEAD);
        expect(r.has_value());
        expect(eq(dev.readRegister("loopback_regs", 0x10), 0xDEADu));
        expect(eq(dev.readRegister("loopback_regs", 0x20), 0u));
    };
};

const boost::ut::suite<"SoapyRaiiWrapper settings"> settingsTests = [] {
    "device-level settings info, write, read"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev   = *result;
        auto  infos = dev.getSettingInfo();
        expect(ge(infos.size(), 2UZ));

        auto r = dev.writeSetting("simulate_timing", "true");
        expect(r.has_value());
        expect(eq(dev.readSetting("simulate_timing"), std::string("true")));
        dev.writeSetting("simulate_timing", "false");
    };

    "per-channel settings info, write, read"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev   = *result;
        auto  infos = dev.getChannelSettingInfo(SOAPY_SDR_RX, 0);
        expect(ge(infos.size(), 2UZ));

        auto r = dev.writeChannelSetting(SOAPY_SDR_RX, 0, "attenuation_dB", "-20");
        expect(r.has_value());
    };
};

const boost::ut::suite<"SoapyRaiiWrapper streaming"> streamTests = [] {
    "stream formats"_test = [] {
        auto result = soapy::Device::make({{"driver", "loopback"}});
        expect(result.has_value());
        auto& dev     = *result;
        auto  formats = dev.getStreamFormats(SOAPY_SDR_RX, 0);
        expect(eq(formats.size(), 3UZ));
    };

    "setupStream, activate, readStream, deactivate"_test = [] {
        auto devResult = soapy::Device::make({{"driver", "loopback"}});
        expect(devResult.has_value());
        if (!devResult.has_value()) {
            return;
        }
        auto& dev = *devResult;
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);

        auto streamResult = dev.setupStream<CF32, SOAPY_SDR_RX>();
        expect(streamResult.has_value()) << "setupStream should succeed";
        if (!streamResult.has_value()) {
            return;
        }
        auto& stream = *streamResult;

        auto r = stream.activate();
        expect(r.has_value());

        // no TX data written — should timeout
        std::vector<CF32> buf(64);
        int               flags  = 0;
        long long         timeNs = 0;
        auto              ret    = stream.readStream(flags, timeNs, static_cast<std::uint32_t>(1000), buf);
        expect(eq(ret, SOAPY_SDR_TIMEOUT));

        r = stream.deactivate();
        expect(r.has_value());
    };

    "TX->RX round-trip through C API wrapper"_test = [] {
        auto devResult = soapy::Device::make({{"driver", "loopback"}});
        expect(devResult.has_value());
        if (!devResult.has_value()) {
            return;
        }
        auto& dev = *devResult;
        dev.setSampleRate(SOAPY_SDR_RX, 0, 1e6);
        dev.setSampleRate(SOAPY_SDR_TX, 0, 1e6);

        // set up TX via raw C API (wrapper doesn't have writeStream yet)
        auto* rawDev   = dev.get();
        auto* txStream = SoapySDRDevice_setupStream(rawDev, SOAPY_SDR_TX, SOAPY_SDR_CF32, nullptr, 0, nullptr);
        expect(neq(txStream, static_cast<SoapySDRStream*>(nullptr)));
        SoapySDRDevice_activateStream(rawDev, txStream, 0, 0, 0);

        // set up RX via wrapper
        auto rxResult = dev.setupStream<CF32, SOAPY_SDR_RX>();
        expect(rxResult.has_value());
        auto& rxStream = *rxResult;
        rxStream.activate();

        // write TX data
        constexpr std::size_t nSamples = 128;
        std::vector<CF32>     txData(nSamples);
        for (std::size_t i = 0UZ; i < nSamples; ++i) {
            txData[i] = {static_cast<float>(i), static_cast<float>(i) * 0.5f};
        }
        const void* txBufs[] = {txData.data()};
        int         txFlags  = 0;
        auto        txRet    = SoapySDRDevice_writeStream(rawDev, txStream, txBufs, nSamples, &txFlags, 0, 100000);
        expect(eq(txRet, static_cast<int>(nSamples)));

        // read RX data via wrapper
        std::vector<CF32> rxData(nSamples);
        int               rxFlags = 0;
        long long         timeNs  = 0;
        auto              rxRet   = rxStream.readStream(rxFlags, timeNs, 100000u, rxData);
        expect(eq(rxRet, static_cast<int>(nSamples)));
        expect(eq(rxData[0], txData[0]));
        expect(eq(rxData[nSamples - 1], txData[nSamples - 1]));

        rxStream.deactivate();
        SoapySDRDevice_deactivateStream(rawDev, txStream, 0, 0);
        SoapySDRDevice_closeStream(rawDev, txStream);
    };
};

int main() { /* not needed for UT */ }
