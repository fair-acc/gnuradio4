#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>

#include <cstdio>
#include <string>

struct PassThrough : gr::Block<PassThrough> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    GR_MAKE_REFLECTABLE(PassThrough, in, out);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x; }
};

int main() {
    gr::Graph         graph;
    auto&             block = graph.emplaceBlock<PassThrough>();
    const std::string name  = block.unique_name;
    std::printf("instantiated %s\n", name.c_str());
    return 0;
}
