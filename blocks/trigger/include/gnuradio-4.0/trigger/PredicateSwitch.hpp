#ifndef GNURADIO_TRIGGER_PREDICATESWITCH_HPP
#define GNURADIO_TRIGGER_PREDICATESWITCH_HPP

#include <algorithm>
#include <concepts>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/algorithm/ExpressionSymbols.hpp>

namespace gr::blocks::trigger {

GR_REGISTER_BLOCK(gr::blocks::trigger::PredicateSwitch, [T], [ float, double ])

template<std::floating_point T>
struct PredicateSwitch : gr::Block<PredicateSwitch<T>, gr::NoTagPropagation> {
    using Description = Doc<R"(@brief route each sample to the output of the first predicate that holds for it, sample-exact

    in      ─1─5─2─7─▶          (no RxMarbles equivalent: Rx partitions a stream in two, this one in N)
    out#0   ───5───7─▶          predicates = "x > 4", "x > 0"
    out#1   ─1───2───▶

A predicate is an exprtk expression over the sample `x`, the block's settings, and the fields of the tags seen so far,
by their short name (`sample_rate` reads `gr:sample_rate`); a field no tag has carried yet is NaN. A sample no predicate
claims is counted and dropped; its tags go with it.
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>                          in;
    std::vector<gr::PortOut<T, gr::Async>> out;

    A<std::vector<std::string>, "predicates", Doc<"one exprtk predicate per output, tried in output order">> predicates;

    A<gr::Size_t, "n unmatched", Doc<"samples no predicate claimed">> n_unmatched = 0U;

    GR_MAKE_REFLECTABLE(PredicateSwitch, in, out, predicates, n_unmatched);

    struct Route {
        gr::algorithm::ExpressionSymbols<T> symbols;
        exprtk::expression<T>               expression;
        bool                                compiled = false;
    };

    T                        _sample = T{};
    std::vector<Route>       _routes;
    std::vector<std::size_t> _published;

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("predicates")) {
            out.resize(predicates.value.size());
            compileRoutes();
        }
    }

    /** exprtk holds references into this block, so a block moved after its settings rebinds here */
    void start() { compileRoutes(); }

    template<gr::OutputSpanLike TOutput>
    gr::work::Status processBulk(gr::InputSpanLike auto& inSpan, std::span<TOutput>& outs) {
        _published.assign(outs.size(), 0UZ);
        auto              tags     = inSpan.rawTags();
        auto              nextTag  = tags.begin();
        std::size_t       consumed = 0UZ;
        const std::size_t routes   = std::min(outs.size(), _routes.size());
        for (; consumed < inSpan.size(); ++consumed) {
            const std::size_t at       = inSpan.streamIndex + consumed;
            auto              tagsHere = nextTag;
            while (nextTag != tags.end() && (*nextTag).index <= at) {
                std::ranges::for_each(_routes, [&](Route& route) { route.symbols.assign(gr::property_map_view{(*nextTag).map}); });
                ++nextTag;
            }

            _sample            = inSpan[consumed];
            std::size_t target = routes;
            for (std::size_t k = 0UZ; k < routes && target == routes; ++k) {
                if (_routes[k].compiled && _routes[k].expression.value() != T{}) {
                    target = k;
                }
            }
            if (target == routes) {
                n_unmatched = n_unmatched + 1U;
                continue;
            }
            if (_published[target] >= outs[target].size()) {
                nextTag = tagsHere;
                break;
            }
            for (auto tag = tagsHere; tag != nextTag; ++tag) {
                outs[target].publishTag((*tag).map, _published[target]);
            }
            outs[target][_published[target]++] = _sample;
        }

        for (std::size_t channel = 0UZ; channel < outs.size(); ++channel) {
            outs[channel].publish(_published[channel]);
        }
        if (!inSpan.consume(consumed)) {
            return gr::work::Status::ERROR;
        }
        return gr::work::Status::OK;
    }

private:
    void compileRoutes() {
        _routes.clear();
        /** exprtk keeps references into each route's symbol table */
        _routes.reserve(predicates.value.size());
        for (const std::string& predicate : predicates.value) {
            Route& route = _routes.emplace_back();
            route.symbols.table.add_variable("x", _sample);
            gr::algorithm::bindSettings(route.symbols.table, *this);
            if (const auto compiled = route.symbols.compile(predicate, route.expression, gr::algorithm::UnknownIdentifiers::readFromTags); compiled) {
                route.compiled = true;
            } else {
                gr::log::warning("PredicateSwitch: predicate '{}' never matches: {}", predicate, compiled.error());
            }
        }
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_PREDICATESWITCH_HPP
