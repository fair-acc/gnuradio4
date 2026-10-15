// expect-file: ok
// naming-check's ground truth: one declaration per rule and per exception of CLAUDE.md §3 as decided on 2026-10-10,
// each marked with what the rule says, `expect: R<n>` or `expect: ok`; `-> name` is the spelling a violation must
// propose, `-> none` a proposal that would clash and so must not be made. Parsed with `--std-style fixture_std`.
#include "Fixture.hpp"
#include "Gauge.hpp"
#include "Unrelated.hpp"
#include "thread_pool.hpp"

#include <array>
#include <cstddef>
#include <utility>

#define GR_MAKE_REFLECTABLE(...)              // expect: ok
#define make_thing(x)            x            // expect: R13
#define forceinline              inline       // expect: ok
#define noreturn                 [[noreturn]] // expect: ok

namespace gr {
template<typename T>
struct Block {};
template<typename T>
struct PortIn {};
template<typename T>
struct PortOut {};
} // namespace gr

namespace boost::ut {
template<typename T = int>
struct suite { // expect: R1
    template<typename TBody>
    suite(TBody) {} // NOLINT: a suite is initialised from its body
};
} // namespace boost::ut

namespace BadSpace { // expect: R4
}

namespace fixture {

// R1 types
struct GoodType {};  // expect: ok
struct bad_type {};  // expect: R1 -> BadType
using index_t = int; // expect: ok
template<typename T>
struct is_widget {}; // expect: ok
struct MemberTypes {
    using value_type   = int; // expect: ok
    using element_type = int; // expect: ok
    using badAlias     = int; // expect: R1 -> BadAlias
};
template<typename T>
using bad_alias_template = T; // expect: R1 -> BadAliasTemplate

// R2 concepts and their template parameters
template<typename T>
concept Drawable = true; // expect: ok
template<typename T>
concept drawable_thing = true; // expect: ok
template<typename T>
concept Bad_Concept = true; // expect: R2
template<typename tBad>     // expect: R11 -> TBad
concept Sized = true;       // expect: ok

// R3 enumerators: one case per enum, the minority reported; a single letter counts for neither
enum class Mode { idle, running };       // expect: ok
enum class State { IDLE, RUNNING_FAST }; // expect: ok
enum class Planet { Earth, Mars };       // expect: ok
enum class Axis { X, Y, horizontal };    // expect: ok
enum class Mixed {
    first,  // expect: ok
    second, // expect: ok
    Third,  // expect: R3 -> third
};
enum class Odd {
    snake_value, // expect: R3
};

// R5 functions
void goodFunction();   // expect: ok
void goodFunction() {} // expect: ok
void Bad_function() {} // expect: R5 -> badFunction
void emplace_back() {} // expect: R5 -> emplaceBack
class Methods {
public:
    void push_back(int) {}      // expect: ok
    void _publicUnderscore() {} // expect: R5
    int  count = 0;             // expect: ok
    void setCount(int count_) { // expect: ok
        count = count_;
    }
    ~Methods() {
        int           Bad_local = 0; // expect: R9 -> badLocal
        constexpr int limit     = 3; // expect: R10 -> kLimit
        static_cast<void>(Bad_local + limit);
    }

private:
    void _helperMethod() {}  // expect: ok
    void _helper_method() {} // expect: ok
};

// R6 a Block's reflected settings; ports lowerCamel
struct MyBlock : gr::Block<MyBlock> {
    gr::PortIn<float>  in;                   // expect: ok
    gr::PortOut<float> outValue;             // expect: ok
    gr::PortIn<float>  secondInput;          // expect: ok
    float              sample_rate   = 1.0f; // expect: ok
    float              gainFactor    = 1.0f; // expect: R6 -> gain_factor
    int                _hostState    = 0;    // expect: ok
    int                _device_state = 0;    // expect: ok

    GR_MAKE_REFLECTABLE(MyBlock, in, outValue, sample_rate, gainFactor, _hostState, _device_state);
};

// R7 public fields other than a Block's settings: lowerCamel
struct Plain {
    int               count         = 0; // expect: ok
    int               itemCount     = 0; // expect: ok
    int               item_total    = 0; // expect: R7 -> itemTotal
    int               _scratchValue = 0; // expect: ok
    static inline int shared_total  = 0; // expect: R7 -> sharedTotal
    static inline int sharedCounter = 0; // expect: ok
};
struct Clash {
    int item_count = 0; // expect: R7 -> none
    int itemCount  = 0; // expect: ok
};
struct Wire {
    int frameIndex = 0; // expect: ok

    GR_MAKE_REFLECTABLE(Wire, frameIndex);
};

// R8 private and protected fields: a leading `_`, the case after it free
class Guarded {
    int _count      = 0; // expect: ok
    int _item_count = 0; // expect: ok
    int itemCount   = 0; // expect: R8 -> _itemCount

public:
    int sum() const { return _count + _item_count + itemCount; }
};

// R9 parameters and locals
int parameters(int goodParam,                                                              // expect: ok
    int            Bad_param,                                                              // expect: R9 -> badParam
    int            N,                                                                      // expect: ok
    int            Nfft,                                                                   // expect: ok
    int            x_0,                                                                    // expect: ok
    int            S_k,                                                                    // expect: ok
    int            sumRe_02,                                                               // expect: ok
    int            TooLongName,                                                            // expect: R9 -> tooLongName
    int            _,                                                                      // expect: ok
    int            long_) {                                                                           // expect: ok
    auto [firstPart, Second_part] = std::pair{goodParam, Bad_param};                       // expect: R9
    int localValue                = N + Nfft + x_0 + S_k + sumRe_02 + TooLongName + long_; // expect: ok
    int i                         = 0;                                                     // expect: ok
    return firstPart + Second_part + localValue + i;
}
struct WithConstructor {
    int value;
    explicit WithConstructor(int value_) : value(value_) {} // expect: ok
};
int                      globalCounter  = 0;     // expect: ok
int                      Global_Counter = 0;     // expect: R9 -> none
const boost::ut::suite<> Any_Name_Tests = [] {}; // expect: ok

// R10 constants: every fixed value, not a closure, not a value computed from template parameters
constexpr int                 kLimit     = 3;                         // expect: ok
constexpr int                 maxCount   = 3;                         // expect: R10 -> kMaxCount
constexpr int                 count      = 3;                         // expect: R10 -> kCount
constexpr int                 M          = 4;                         // expect: ok
const int                     kOther     = 2;                         // expect: ok
const int                     otherLimit = 2;                         // expect: R10 -> kOtherLimit
const std::array<int, 2>      limits{1, 2};                           // expect: R10 -> kLimits
const std::array<GoodType, 2> table{};                                // expect: ok
constexpr auto                squareOf = [](int v) { return v * v; }; // expect: ok
template<typename T>
constexpr bool is_small_v = sizeof(T) < 4; // expect: ok
template<typename T>
struct Sizes {
    static constexpr std::size_t bytes      = sizeof(T); // expect: ok
    static constexpr int         fixedValue = 4;         // expect: R10 -> kFixedValue
};
int constants() {
    constexpr int local = 2; // expect: R10 -> kLocal
    return local + kLimit + maxCount + count + M + kOther + otherLimit + limits[0] + squareOf(2) + Sizes<int>::bytes + Global_Counter;
}

// R11 type template parameters, R12 non-type template parameters
template<typename T,      // expect: ok
    typename TValue,      // expect: ok
    typename value_t,     // expect: R11
    std::size_t nPorts,   // expect: ok
    int         Count,    // expect: ok
    int         MAX_SIZE, // expect: ok
    int         bad_param>        // expect: R12
struct Templated {};

} // namespace fixture

// std-style scope (`--std-style fixture_std`): std naming throughout
namespace fixture_std {
struct small_vector {      // expect: ok
    using size_type = int; // expect: ok
    int  size_      = 0;   // expect: ok
    void push_all() {}     // expect: ok
};
struct CamelInStd {}; // expect: R1 -> camel_in_std
} // namespace fixture_std

int main() { return fixture::parameters(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) + fixture::constants() + fixture::GaugeFace{}.size + fixture::GaugeNeedle{}.size + fixture_std::thread_pool_size() + fixture::fixtureEntry(); }
