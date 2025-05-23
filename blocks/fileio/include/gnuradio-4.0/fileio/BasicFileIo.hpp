#ifndef BASICFILEIO_HPP
#define BASICFILEIO_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <magic_enum.hpp>

#include <chrono>
#include <complex>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>

namespace gr::blocks::fileio {

namespace detail {

inline void ensureDirectoryExists(const std::filesystem::path& filePath) { std::filesystem::create_directories(filePath.parent_path()); }

inline std::vector<std::filesystem::path> getSortedFilesContaining(const std::string& fileName) {
    std::filesystem::path filePath(fileName);
    if (!std::filesystem::exists(filePath.parent_path())) {
        throw gr::exception(std::format("path/file '{}' does not exist.", fileName));
    }

    std::vector<std::filesystem::path> matchingFiles;
    std::copy_if(std::filesystem::directory_iterator(filePath.parent_path()), std::filesystem::directory_iterator{}, std::back_inserter(matchingFiles), //
        [&](const auto& entry) { return entry.is_regular_file() && entry.path().string().find(filePath.filename().string()) != std::string::npos; });

    std::sort(matchingFiles.begin(), matchingFiles.end());
    return matchingFiles;
}

[[nodiscard]] inline std::uintmax_t getFileSize(const std::filesystem::path& filePath) {
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(std::format("file '{}' does not exist.", filePath.string()));
    }
    return std::filesystem::file_size(filePath);
}

[[maybe_unused]] inline std::vector<std::string> deleteFilesContaining(const std::string& fileName) {
    std::filesystem::path filePath(fileName);
    if (!std::filesystem::exists(filePath.parent_path())) {
        return {};
    }

    std::vector<std::string> deletedFiles;
    for (const auto& entry : std::filesystem::directory_iterator(filePath.parent_path())) {
        if (entry.is_regular_file() && entry.path().string().find(filePath.filename().string()) != std::string::npos) {
            deletedFiles.push_back(entry.path().string());
            std::filesystem::remove(entry.path());
        }
    }

    return deletedFiles;
}

} // namespace detail

enum class Mode { overwrite, append, multi };

GR_REGISTER_BLOCK(gr::blocks::fileio::BasicFileSink, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])

template<typename T>
struct BasicFileSink : Block<BasicFileSink<T>> {
    using Description = Doc<R""(A sink block for writing a stream to a binary file.
The file can be played back using a 'BasicFileSource' or read by any program that supports binary files (e.g. Python, C, C++, MATLAB).
For complex types, the binary file contains [float, double]s in IQIQIQ order. No metadata is included with the binary data.)
Important: this implementation assumes a host-order, CPU architecture specific byte order!)"">;
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>; // optional shortening

    PortIn<T> in;

    A<std::string, "file name", Doc<"base filename, prefixed if ">, Visible>              file_name;
    A<Mode, "mode", Doc<"mode: \"overwrite\", \"append\", \"multi\"">, Visible>           mode               = Mode::overwrite;
    A<gr::Size_t, "max bytes per file", Doc<"max bytes per file, 0: infinite ">, Visible> max_bytes_per_file = 0U;

    GR_MAKE_REFLECTABLE(BasicFileSink, in, file_name, mode, max_bytes_per_file);

    std::size_t   _totalBytesWritten{0UZ};
    std::size_t   _totalBytesWrittenFile{0UZ};
    std::ofstream _file;
    std::size_t   _fileCounter{0UZ};
    std::string   _actualFileName;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& /*newSettings*/) {
        if (lifecycle::isActive(this->state())) {
            closeFile();
            openNextFile();
        }
    }

    void start() {
        _totalBytesWritten = 0UZ;
        openNextFile();
    }

    void stop() { closeFile(); }

    [[nodiscard]] constexpr work::Status processBulk(InputSpanLike auto& dataIn) {
        if (max_bytes_per_file.value != 0U && _totalBytesWrittenFile >= max_bytes_per_file.value) {
            closeFile();
            openNextFile();
        }

        std::size_t nBytesMax = dataIn.size() * sizeof(T);
        if (max_bytes_per_file.value != 0U) {
            nBytesMax = std::min(nBytesMax, static_cast<std::size_t>(max_bytes_per_file.value) - _totalBytesWrittenFile);
        }
        _file.write(reinterpret_cast<const char*>(dataIn.data()), static_cast<std::streamsize>(nBytesMax));
        if (!dataIn.consume(nBytesMax / sizeof(T))) {
            throw gr::exception("could not consume input samples");
        }

        if (!_file) {
            throw gr::exception(std::format("failed to write to file '{}'.", _actualFileName));
        }

        _totalBytesWritten += nBytesMax;
        _totalBytesWrittenFile += nBytesMax;

        return work::Status::OK;
    }

private:
    void closeFile() {
        if (_file.is_open()) {
            _file.close();
        }
    }
    void openNextFile() {
        closeFile();
        _totalBytesWrittenFile = 0UZ;

        detail::ensureDirectoryExists(file_name.value);

        std::filesystem::path filePath(file_name.value);
        if (!std::filesystem::exists(filePath.parent_path())) {
            throw gr::exception(std::format("path/file '{}' does not exist.", file_name.value));
        }

        // Open file handle based on mode
        switch (mode) {
        case Mode::overwrite: {
            _actualFileName = file_name.value;
            _file.open(_actualFileName, std::ios::binary | std::ios::trunc);
        } break;
        case Mode::append: {
            _actualFileName = file_name.value;
            _file.open(_actualFileName, std::ios::binary | std::ios::app);
        } break;
        case Mode::multi: {
            // _fileCounter ensures that the filenames are unique and still sortable by date-time, with an additional counter to handle rapid successive file creation.
            _actualFileName = filePath.parent_path() / (gr::time::getIsoTime() + "_" + std::to_string(_fileCounter++) + "_" + filePath.filename().string());
            _file.open(_actualFileName, std::ios::binary);
            break;
        }
        default: throw gr::exception("unsupported file mode.");
        }

        if (!_file) {
            throw gr::exception(std::format("failed to open file '{}'.", _actualFileName));
        }
    }
};

GR_REGISTER_BLOCK(gr::blocks::fileio::BasicFileSource, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])

template<typename T>
struct BasicFileSource : Block<BasicFileSource<T>> {
    using Description = Doc<R""(A source block for reading a binary file and outputting the data.
This source is the counterpart to 'BasicFileSink'.
For complex types, the binary file contains [float, double]s in IQIQIQ order. No metadata is expected in the binary data.
Important: this implementation assumes a host-order, CPU architecture specific byte order!)"">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>; // optional shortening

    PortOut<T> out;

    A<std::string, "file name", Doc<"Base filename, prefixed if necessary">, Visible>          file_name;
    A<Mode, "mode", Doc<"mode: \"overwrite\", \"append\", \"multi\"">, Visible>                mode         = Mode::overwrite;
    A<bool, "repeat", Doc<"true: repeat back-to-back">>                                        repeat       = false;
    A<gr::Size_t, "offset", Doc<"file start offset in bytes">, Visible>                        offset       = 0U;
    A<gr::Size_t, "length", Doc<"max number of samples items to read (0: infinite)">, Visible> length       = 0U;
    A<std::string, "trigger name", Doc<"name of trigger added to each file chunk">>            trigger_name = "BasicFileSource::start";

    GR_MAKE_REFLECTABLE(BasicFileSource, out, file_name, mode, repeat, offset, length, trigger_name);

    std::ifstream                      _file;
    std::vector<std::filesystem::path> _filesToRead;
    bool                               _emittedStartTrigger = false;
    std::size_t                        _totalBytesRead      = 0UZ;
    std::size_t                        _totalBytesReadFile  = 0UZ;
    std::size_t                        _currentFileIndex    = 0UZ;
    std::string                        _currentFileName;

    void start() {
        _currentFileIndex = 0UZ;
        _totalBytesRead   = 0UZ;
        _filesToRead.clear();

        std::filesystem::path filePath(file_name.value);
        if (!std::filesystem::exists(filePath.parent_path())) {
            throw gr::exception(std::format("path/file '{}' does not exist.", file_name.value));
        }

        switch (mode) {
        case Mode::overwrite:
        case Mode::append: {
            _filesToRead.push_back(filePath);
        } break;
        case Mode::multi: {
            _filesToRead = detail::getSortedFilesContaining(file_name.value);
        } break;
        default: throw gr::exception("unsupported file mode.");
        }

        openNextFile();
    }

    void stop() { closeFile(); }

    [[nodiscard]] constexpr work::Status processBulk(OutputSpanLike auto& dataOut) noexcept {
        if (!_file.is_open()) {
            return work::Status::DONE;
        }
        std::size_t nOutAvailable = dataOut.size() * sizeof(T);
        if (length.value != 0U) {
            nOutAvailable = std::min(nOutAvailable, (length.value * sizeof(T) - _totalBytesReadFile));
        }

        std::size_t bytesRead = static_cast<std::size_t>(_file.read(reinterpret_cast<char*>(dataOut.data()), static_cast<std::streamsize>(nOutAvailable)).gcount());
        if (!_emittedStartTrigger && !trigger_name.value.empty()) {
            dataOut.publishTag(
                property_map{
                    {std::string(tag::TRIGGER_NAME.shortKey()), trigger_name.value},                                                     //
                    {std::string(tag::TRIGGER_TIME.shortKey()), settings::convertTimePointToUint64Ns(std::chrono::system_clock::now())}, //
                    {std::string(tag::TRIGGER_OFFSET.shortKey()), 0.f}                                                                   //
                },
                0UZ);
            _emittedStartTrigger = true;
        }

        dataOut.publish(bytesRead / sizeof(T));
        _totalBytesRead += bytesRead;
        _totalBytesReadFile += bytesRead;

        if (bytesRead < nOutAvailable || (length.value != 0U && (_totalBytesReadFile >= length.value * sizeof(T)))) {
            closeFile();
            if (_currentFileIndex < _filesToRead.size()) {
                openNextFile();
                return work::Status::OK;
            } else if (repeat) {
                _currentFileIndex = 0UZ;
                openNextFile();
                return work::Status::OK;
            } else {
                return work::Status::DONE;
            }
        }

        return work::Status::OK;
    }

private:
    void closeFile() {
        if (_file.is_open()) {
            _file.close();
        }
    }
    void openNextFile() {
        if (_currentFileIndex >= _filesToRead.size()) {
            return;
        }
        _totalBytesReadFile  = 0UZ;
        _emittedStartTrigger = false;

        _currentFileName = _filesToRead[_currentFileIndex].string();
        _file.open(_currentFileName, std::ios::binary);
        if (!_file) {
            throw gr::exception(std::format("failed to open file '{}'.", _currentFileName));
        }
        if (offset.value != 0U) {
            _file.seekg(offset.value * sizeof(T), std::ios::beg);
        }
        _currentFileIndex++;
    }
};

} // namespace gr::blocks::fileio

#endif // BASICFILEIO_HPP
