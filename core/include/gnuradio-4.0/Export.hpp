#ifndef GNURADIO_EXPORT_MACRO_HPP
#define GNURADIO_EXPORT_MACRO_HPP

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_MYPLUGIN
#define GNURADIO_EXPORT __declspec(dllexport)
#else
#define GNURADIO_EXPORT __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define GNURADIO_EXPORT __attribute__((visibility("default")))
#else
#define GNURADIO_EXPORT
#endif
#endif

#endif
