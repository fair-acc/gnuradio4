//import boost.ut;
#include <boost/ut.hpp>

#include <fmt/ranges.h>

#include "buffer.hpp"
#include "graph_contracts.h"
#include "refl.hpp"

const boost::ut::suite PortApiTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace fair;

    "PortApi"_test = [] {
        static_assert(Port<IN<float, "in">>);
        static_assert(Port<decltype(IN<float>("in"))>);
        static_assert(Port<OUT<float, "out">>);
        static_assert(Port<IN_MSG<float, "in_msg">>);
        static_assert(Port<OUT_MSG<float, "out_msg">>);

        static_assert(IN<float, "in">::static_name() == fixed_string("in"));
        static_assert(requires { IN<float>("in").name(); });
    };

    "PortBufferApi"_test = [] {
        OUT<float, "out0"> output_port;
        BufferWriter auto& writer = output_port.writer();
        expect(ge(writer.available(), 32));

        IN<float, "int0"> input_port;
        BufferReader auto& reader = input_port.reader();
        expect(eq(reader.available(), 0));
        input_port.setBuffer(output_port.get_buffer());

        expect(eq(output_port.get_buffer().n_readers(), 1));

        int offset = 1;
        auto lambda = [&offset](auto& w) {
            std::iota(w.begin(), w.end(), offset);
            fmt::print("generated output vector: {}\n", w);
            offset += w.size();
        };

        writer.publish(lambda, 32);
        expect(writer.try_publish(lambda, 32));
    };

    "RuntimePortApi"_test = [] {
        std::vector<dyn_port> port_list;

        port_list.emplace_back(OUT<float, "out">());
        port_list.emplace_back(IN<float, "in">());

        for (int i=0; i < 3; i++) {
            port_list.emplace_back(IN<float>(fmt::format("in{}", i)));
        }
        expect(eq(port_list.size(), 5));
    };
};


int main() {}
