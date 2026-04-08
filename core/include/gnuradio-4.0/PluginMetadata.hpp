#ifndef GNURADIO_PLUGIN_METADATA_HPP
#define GNURADIO_PLUGIN_METADATA_HPP

#include <string>

#include <gnuradio-4.0/Export.hpp>

struct GNURADIO_EXPORT gr_plugin_metadata {
    std::string plugin_name;
    std::string plugin_author;
    std::string plugin_license;
    std::string plugin_version;
    std::string block_type = {};
};

#endif // GNURADIO_PLUGIN_METADATA_HPP
