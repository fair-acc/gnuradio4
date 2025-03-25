#ifndef GR_BLOCKLIB_INIT_HPP

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_MYPLUGIN
#define GNURADIO4_EXPORT __declspec(dllexport)
#else
#define GNURADIO4_EXPORT __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define GNURADIO4_EXPORT __attribute__((visibility("default")))
#else
#define GNURADIO4_EXPORT
#endif
#endif

namespace gr {
class BlockRegistry;
}

GNURADIO4_EXPORT gr::BlockRegistry& grBlockLibRegistry();

#endif
