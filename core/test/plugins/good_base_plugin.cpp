#include "good_base_plugin.hpp"

GR_PLUGIN("Good Base Plugin", "Unknown", "MIT", "v1")

namespace bts = gr::traits::block;

static_assert(bts::all_input_ports<good::cout_sink<float>>::size == 1);
static_assert(std::is_same_v<bts::all_input_port_types<good::cout_sink<float>>, gr::meta::typelist<float>>);
static_assert(bts::stream_input_ports<good::cout_sink<float>>::size == 1);
static_assert(std::is_same_v<bts::stream_input_port_types<good::cout_sink<float>>, gr::meta::typelist<float>>);

static_assert(bts::all_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::all_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);
static_assert(bts::stream_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::stream_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);

static_assert(bts::all_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::all_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);
static_assert(bts::stream_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::stream_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);

static_assert(bts::all_input_ports<good::fixed_source<float>>::size == 0);
static_assert(std::is_same_v<bts::all_input_port_types<good::fixed_source<float>>, gr::meta::typelist<>>);
static_assert(bts::stream_input_ports<good::fixed_source<float>>::size == 0);
static_assert(std::is_same_v<bts::stream_input_port_types<good::fixed_source<float>>, gr::meta::typelist<>>);
static_assert(bts::all_output_ports<good::fixed_source<float>>::size == 1);
static_assert(std::is_same_v<bts::all_output_port_types<good::fixed_source<float>>, gr::meta::typelist<float>>);
static_assert(bts::stream_output_ports<good::fixed_source<float>>::size == 1);
static_assert(std::is_same_v<bts::stream_output_port_types<good::fixed_source<float>>, gr::meta::typelist<float>>);
