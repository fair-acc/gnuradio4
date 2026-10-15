// devtools/code-metrics' fixture: each `…Split` function is written again as `…Inlined`, its single-use callees inlined
// by hand at their call depth, so clang-tidy's score of the twin is the ground truth for the split-invariant figure;
// the ambient-reach expectations are listed in check-fixture.sh

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <span>
#include <string>
#include <vector>

namespace ImGui {
double GetTime();
}

int gCounter = 0;

const int kLimit = 3;

int depthZeroPart(int count) {
    int even = 0;
    for (int index = 0; index < count; ++index) {
        if (index % 2 == 0) {
            ++even;
        }
    }
    return even;
}

int depthZeroSplit(bool wanted, int count) {
    const int even = depthZeroPart(count);
    if (wanted) {
        return even;
    }
    return 0;
}

int depthZeroInlined(bool wanted, int count) {
    int even = 0;
    for (int index = 0; index < count; ++index) {
        if (index % 2 == 0) {
            ++even;
        }
    }
    if (wanted) {
        return even;
    }
    return 0;
}

int depthTwoPart(int value) {
    const auto doubled = [](int positive) {
        if (positive > 0) {
            return 2 * positive;
        }
        return 0;
    };
    if (value > 10 && value < 20) {
        return doubled(value);
    } else if (value < 0) {
        return value > -5 ? 1 : 2;
    }
    return 0;
}

int depthTwoSplit(int count) {
    int total = 0;
    for (int index = 0; index < count; ++index) {
        if (index > 2) {
            total += depthTwoPart(index);
        }
    }
    return total;
}

int depthTwoInlined(int count) {
    int total = 0;
    for (int index = 0; index < count; ++index) {
        if (index > 2) {
            const auto doubled = [](int positive) {
                if (positive > 0) {
                    return 2 * positive;
                }
                return 0;
            };
            int part = 0;
            if (index > 10 && index < 20) {
                part = doubled(index);
            } else if (index < 0) {
                part = index > -5 ? 1 : 2;
            }
            total += part;
        }
    }
    return total;
}

int chainLeaf(int value) {
    while (value > 1) {
        value /= 2;
    }
    return value;
}

int chainMiddle(int value) {
    if (value > 0) {
        return chainLeaf(value);
    }
    return 0;
}

int chainSplit(bool wanted, int value) {
    if (wanted) {
        return chainMiddle(value);
    }
    return 0;
}

int chainInlined(bool wanted, int value) {
    if (wanted) {
        if (value > 0) {
            while (value > 1) {
                value /= 2;
            }
            return value;
        }
        return 0;
    }
    return 0;
}

int wrapperLeaf(int value) {
    while (value > 1) {
        value /= 2;
    }
    return value;
}

int wrapperPassThrough(int value) { return wrapperLeaf(value); }

int wrapperSplit(int count) {
    int total = 0;
    for (int index = 0; index < count; ++index) {
        if (index > 2) {
            total += wrapperPassThrough(index);
        }
    }
    return total;
}

int wrapperInlined(int count) {
    int total = 0;
    for (int index = 0; index < count; ++index) {
        if (index > 2) {
            int value = index;
            while (value > 1) {
                value /= 2;
            }
            total += value;
        }
    }
    return total;
}

int sharedPart(int value) {
    if (value > 0) {
        return 1;
    }
    return 0;
}

int twoCallersFirst(bool wanted, int value) {
    if (wanted) {
        return sharedPart(value);
    }
    return 0;
}

int twoCallersSecond(int value) { return sharedPart(value) + 1; }

int testedPart(int value) {
    if (value > 0) {
        return 1;
    }
    return 0;
}

int testedCaller(bool wanted, int value) {
    if (wanted) {
        return testedPart(value);
    }
    return 0;
}

double drawingClock() { return ImGui::GetTime(); }

double frameStart() { return drawingClock() * 2.0; }

long long wallClock() { return std::chrono::steady_clock::now().time_since_epoch().count(); }

const char* homeDirectory() { return std::getenv("HOME"); }

int bumpCounter() { return ++gCounter; }

int nextIdentifier() {
    static int next = 0;
    return ++next;
}

struct CountOnExit {
    ~CountOnExit() { ++gCounter; }
};

int counterPlusOne() { return bumpCounter() + 1; }

int counterThroughTwoCalls() { return counterPlusOne(); }

int limited(int value) { return std::min(value, kLimit); }

void sortInPlace(std::vector<int>& values) { std::ranges::sort(values); }

bool fileExists(const std::string& path) { return std::filesystem::exists(path); }

std::string stemOf(const std::string& path) { return std::filesystem::path{path}.stem().string(); }

void reportCount(int count) { std::println(stderr, "{}", count); }

// unscoped pairs, weighted by hand: 1 per matched pair, +1 per exit leaving it, +1 when opener and closer sit in different branches
struct ItemStack {
    void PushItem(int) {}
    void PopItem() {}
};

void pairBalanced(ItemStack& stack) { // 1
    stack.PushItem(1);
    stack.PopItem();
}

void pairReturnBetween(ItemStack& stack, bool skip) { // 2: the return leaves the pushed item behind
    stack.PushItem(1);
    if (skip) {
        return;
    }
    stack.PopItem();
}

void pairLoopInside(ItemStack& stack, std::span<const int> values) { // 1: the continue stays inside its own loop
    stack.PushItem(1);
    for (const int value : values) {
        if (value < 0) {
            continue;
        }
    }
    stack.PopItem();
}

void pairContinueSkips(ItemStack& stack, std::span<const int> values) { // 2: the continue skips the pop
    for (const int value : values) {
        stack.PushItem(value);
        if (value < 0) {
            continue;
        }
        stack.PopItem();
    }
}

void pairConditionalPush(ItemStack& stack, bool clip) { // 2: pushed in one branch, popped in another
    if (clip) {
        stack.PushItem(1);
    }
    stack.PopItem();
}

void pairPoppedEarly(ItemStack& stack, bool done) { // 2: the first pop sits in a branch; the second closes nothing here
    stack.PushItem(1);
    if (done) {
        stack.PopItem();
        return;
    }
    stack.PopItem();
}

struct ScopedItem { // 0: opened by the constructor, closed by the destructor, never in one function
    ItemStack& stack;
    explicit ScopedItem(ItemStack& into) : stack(into) { stack.PushItem(1); }
    ScopedItem(const ScopedItem&)            = delete;
    ScopedItem& operator=(const ScopedItem&) = delete;
    ~ScopedItem() { stack.PopItem(); }
};

void pairScoped(ItemStack& stack) { const ScopedItem item{stack}; }

void pushBackOnly(std::vector<int>& values) { // 0: push_back is not an opener
    values.push_back(1);
    values.pop_back();
}

// repeated expressions: the doubled first value written three times, its `values.front()*2+1` part only inside it, and a pair written twice
int repeatedThrice(const std::vector<int>& values) {
    const int first  = values.empty() ? 0 : values.front() * 2 + 1;
    const int second = values.empty() ? 0 : values.front() * 2 + 1;
    const int twice  = values.size() > 1 ? values.back() : 0;
    const int again  = values.size() > 1 ? values.back() : 0;
    return first + second + twice + again + (values.empty() ? 0 : values.front() * 2 + 1);
}

// same argument everywhere: factor is 3 at both calls of scaledBy; varyingFactor's is not; onceCalled has a single call
int scaledBy(int value, int factor) { return value * factor; }
int varyingFactor(int value, int factor) { return value + factor; }
int onceCalled(int factor) { return factor * 2; }

int scaledTwice(int value) { return scaledBy(value, 3) + scaledBy(value + 1, 3) + varyingFactor(value, 2) + varyingFactor(value, 5) + onceCalled(4); }

// write-only fields: written is set by an initialiser and an assignment and never read; counted is read by +=; shown is read;
// marked, in an unnamed struct, is written only
struct Tally {
    int written = 0;
    int counted = 0;
    int shown   = 0;
    struct {
        bool marked = false;
    } flags;
};

int tallied(int value) {
    Tally tally{.written = 1, .counted = 0, .shown = value};
    tally.written      = value;
    tally.flags.marked = true;
    tally.counted += value;
    return tally.shown + tally.counted;
}

struct Compared { // read through the defaulted ==, so not listed
    int  value                             = 0;
    bool operator==(const Compared&) const = default;
};

bool sameCompared(int left, int right) { return Compared{.value = left} == Compared{.value = right}; }

// unused functions: neverCalled is named nowhere; passedAlong is only passed on as a callback
int neverCalled(int value) { return value + 1; }
int passedAlong(int value) { return value - 1; }
int applied(int value, int (*step)(int)) { return step(value); }
int appliedOnce(int value) { return applied(value, passedAlong); }

// flag arguments: paint is a bare true at one call and false at another; the third names it
int drawn(int value, bool paint) { return paint ? value : 0; }

int drawnThrice(int value) {
    const bool measuring = false;
    return drawn(value, true) + drawn(value, false) + drawn(value, measuring);
}

// nested ternaries: one ?: inside another, and the same with the inner choice named
float chosenNested(bool heading, int level) { return heading ? (level == 1 ? 36.0f : 24.0f) : 18.0f; }

float chosenNamed(bool heading, int level) {
    const float headingSize = level == 1 ? 36.0f : 24.0f;
    return heading ? headingSize : 18.0f;
}
