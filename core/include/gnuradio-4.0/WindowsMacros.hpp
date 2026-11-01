#ifndef GNURADIO_WINDOWSMACROS_HPP
#define GNURADIO_WINDOWSMACROS_HPP

// windows.h — included directly or transitively through cpr, curl, httplib, exprtk — defines macros whose spelling
// collides with GR4 identifiers. Include this header after every such include.
#ifdef _WIN32
#undef ERROR     // wingdi.h: 0, collides with gr::work::Status::ERROR
#undef far       // windef.h: empty, collides with gr::algorithm::Field::far
#undef near      // windef.h: empty, collides with gr::algorithm::Field::near
#undef interface // basetyps.h: struct, collides with any identifier of that name
#endif

#endif // GNURADIO_WINDOWSMACROS_HPP
