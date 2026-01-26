# `gr::Drawable<>` & `UICategory` User-API Reference

## Overview

GR4 blocks can contribute to user interfaces through the `Drawable` annotation.
This enables blocks to render visualizations, controls, and status information while remaining toolkit-agnostic.

> **Design Philosophy:** The framework defines _what_ a block wants to display (semantic category), not _how_ it
> renders (toolkit-specific).<br>
> Applications select a UI toolkit at build time; the core remains free of toolkit dependencies.

**Key characteristics:**

- **Semantic categorization** — `UICategory` describes placement intent (toolbar, panel, content area), not layout
  mechanics
- **Separation of concerns** — `processBulk()`/`processOne()` handles data; `draw()` handles presentation
- **Async port integration** — UI blocks emit settings via `PortOut<T, Async>`, leveraging existing infrastructure
- **Toolkit independence** — `Drawable<Category, "toolkit">` allows compile-time toolkit selection

## TLDR; Examples

### Minimal Drawable Block

```cpp
#include <gnuradio-4.0/Block.hpp>

template<typename T>
requires std::is_arithmetic_v<T>
struct PlotSink : gr::Block<PlotSink<T>, gr::Drawable<gr::UICategory::Content, "ImGui">> {
    template<typename U, gr::meta::fixed_string description = "", typename... Args>
    using A = gr::Annotated<U, description, Args...>;

    gr::PortIn<T> in;

    A<gr::Size_t, "history length", gr::Doc<"samples retained in ring buffer">> n_history = 1000UZ;

    GR_MAKE_REFLECTABLE(PlotSink, in, n_history);

    mutable std::mutex  _drawMutex; // guards _history between process*() and draw()
    std::vector<T>      _history;

    void settingsChanged(const gr::property_map& /*oldSettings*/, gr::property_map& newSettings) {
        if (newSettings.contains("n_history")) {
            std::lock_guard lock(_drawMutex);
            _history.reserve(n_history);
        }
    }

    constexpr void processOne(T sample) noexcept {
        std::lock_guard lock(_drawMutex);
        _history.push_back(sample);
        if (_history.size() > n_history) {
            _history.erase(_history.begin());
        }
    }

    gr::work::Status draw(const gr::property_map& /*config*/ = {}) noexcept {
        std::lock_guard lock(_drawMutex);
        renderPlot(_history); // toolkit-specific (mock)
        return gr::work::Status::OK;
    }
};
```

### UI Control Block (emits settings)

```cpp
template<typename T>
requires std::is_arithmetic_v<T>
struct SliderControl : gr::Block<SliderControl<T>, gr::Drawable<gr::UICategory::Toolbar, "ImGui">> {
    template<typename U, gr::meta::fixed_string description = "", typename... Args>
    using A = gr::Annotated<U, description, Args...>;

    gr::PortOut<gr::property_map, gr::Async> out;

    A<std::string, "target key", gr::Doc<"settings key to emit (e.g. 'sample_rate', 'gain')">> target_key = "gain";
    A<T, "value">     value     = T{0.5};
    A<T, "min value"> min_value = T{0};
    A<T, "max value"> max_value = T{1};

    GR_MAKE_REFLECTABLE(SliderControl, out, target_key, value, min_value, max_value);

    gr::work::Status draw(const gr::property_map& /*config*/ = {}) noexcept {
        T new_value = value;
        if (renderSlider(min_value, max_value, new_value)) { // toolkit-specific, returns true if changed
            value = new_value;
            out.publish(gr::property_map{{target_key, value}});
        }
        return gr::work::Status::OK;
    }
};
```

The `target_key` property allows runtime configuration of which downstream block setting the slider controls. For
example, connecting a `SliderControl<float>` with `target_key = "cutoff_freq"` to a filter block automatically binds
emitted values to the filter's `cutoff_freq` setting.

## UICategory Enum

Defines the semantic placement intent for drawable blocks:

| Category       | Purpose                                             | Examples                       |
|----------------|-----------------------------------------------------|--------------------------------|
| `None`         | No UI contribution (default)                        | Processing-only blocks         |
| `MenuBar`      | Global application menu items                       | File/Edit/View menus           |
| `Toolbar`      | Compact, frequently used actions and controls       | Play/pause, sliders, toggles   |
| `StatusBar`    | Always-visible, low-interaction status readouts     | Sample rate, buffer levels     |
| `Content`      | Primary viewport output                             | Plots, charts, canvases        |
| `Panel`        | Secondary panels for detailed interaction           | Inspectors, settings, logs     |
| `Overlay`      | Layered HUD elements over Content                   | Cursors, markers, annotations  |
| `ContextMenu`  | Right-click / long-press contextual popup menus     | Per-item actions               |
| `Dialog`       | Focused workflows (non-modal, consistent placement) | Export, configuration, confirm |
| `Notification` | Transient non-modal feedback                        | Toasts, banners, alerts        |

### Category Selection Guidelines

- **Toolbar** for controls that emit values (sliders, buttons, toggles)
- **Panel** for forms, property editors, detailed settings
- **Content** for data visualization (plots, spectrograms, constellations)
- **Overlay** for annotations that augment Content without replacing it
- **Dialog** for focused multi-step workflows; always non-modal, consistent screen position

> **Note:** Docking, tabbing, and window arrangement are toolkit concerns, not category concerns. A `Panel` block
> declares "I am secondary content" — whether panels dock or tab is decided by the application/toolkit layer.

## Drawable Annotation

```cpp
template<UICategory category_, gr::meta::fixed_string toolkit_ = "">
struct Drawable {
    static constexpr UICategory             kCategory = category_;
    static constexpr gr::meta::fixed_string kToolkit  = toolkit_;
};
```

### Template Parameters

| Parameter   | Type                     | Description                                                      |
|-------------|--------------------------|------------------------------------------------------------------|
| `category_` | `UICategory`             | Semantic placement intent                                        |
| `toolkit_`  | `gr::meta::fixed_string` | Target toolkit identifier (e.g., `"ImGui"`, `"Qt"`, `"console"`) |

### Usage

```cpp
// Content block for ImGui
struct MyChart : Block<MyChart, Drawable<UICategory::Content, "ImGui">> { ... };

// Toolbar control for Qt
struct MySlider : Block<MySlider, Drawable<UICategory::Toolbar, "Qt">> { ... };

// Console-based output (no GUI toolkit)
struct ConsoleSink : Block<ConsoleSink, Drawable<UICategory::Content, "console">> { ... };

// Toolkit-agnostic (empty string — application decides)
struct GenericPanel : Block<GenericPanel, Drawable<UICategory::Panel>> { ... };
```

### Compile-Time Introspection

```cpp
// Concept check
static_assert(gr::IsDrawable<Drawable<UICategory::Content, "ImGui">>);
static_assert(!gr::IsDrawable<int>);

// Access category and toolkit from block type
using MyBlock = MyChart;
constexpr auto category = MyBlock::DrawableControl::kCategory;  // UICategory::Content
constexpr auto toolkit  = MyBlock::DrawableControl::kToolkit;   // "ImGui"
```

## Block Protocol

### Required: `draw()` Method

Blocks annotated with `Drawable<Category, Toolkit>` (where `Category != None`) must implement:

```cpp
gr::work::Status draw(const gr::property_map& config = {}) noexcept;
```

**Parameters:**

- `config` — Toolkit-specific configuration passed by the render loop. Contents are an **API contract between the
  application and its blocks** — the framework does not define required keys.

**Return value:**

| Status                | Meaning                           |
|-----------------------|-----------------------------------|
| `work::Status::OK`    | Rendered successfully             |
| `work::Status::DONE`  | Block finished, UI can be removed |
| `work::Status::ERROR` | Rendering failed                  |

### Optional: Layout Hints via `ui_constraints`

Blocks can store toolkit-specific layout preferences in the `ui_constraints` property:

```cpp
struct SettingsPanel : gr::Block<SettingsPanel, gr::Drawable<gr::UICategory::Panel, "ImGui">> {
    // inherited from Block<>: A<property_map, "ui-constraints"> ui_constraints;

    GR_MAKE_REFLECTABLE(SettingsPanel);

    void start() {
        ui_constraints.value = {
            {"preferred_width", 300},
            {"preferred_height", 200},
            {"dockable", true}
        };
    }

    gr::work::Status draw(const gr::property_map& /*config*/ = {}) noexcept {
        renderSettingsPanel(); // toolkit-specific
        return gr::work::Status::OK;
    }
};
```

Interpretation of these hints is entirely toolkit-specific.

### Runtime Category Query

The `BlockModel` interface exposes category for runtime discovery:

```cpp
// From BlockModel (type-erased interface)
[[nodiscard]] virtual UICategory uiCategory() const;

// Usage: find all Content blocks in a graph
for (auto& block : graph.blocks()) {
    if (block->uiCategory() == UICategory::Content) {
        // Register with Content renderer
    }
}
```

## Data Flow Patterns

### Pattern 1: Visualization Sink (data in, display out)

```
┌──────────────┐         ┌─────────────────┐
│ SignalSource │──data──▶│ PlotSink        │ (Content)
└──────────────┘         │ draw() renders  │
                         └─────────────────┘
```

The sink's `processOne()`/`processBulk()` accumulates data; `draw()` visualizes it.

### Pattern 2: Control Source (user input out)

```
┌─────────────────┐              ┌──────────────┐
│ SliderControl   │──property───▶│ FilterBlock  │
│ draw() captures │   _map       │ gain setting │
└─────────────────┘              └──────────────┘
```

The control's `draw()` captures user input and publishes via async output port.

### Pattern 3: Bidirectional (display + control)

```
┌──────────────┐    data     ┌─────────────────┐
│ FilterBlock  │────────────▶│ FilterPanel     │ (Panel)
│              │◀────────────│ draw() shows    │
│              │  settings   │ params + sliders│
└──────────────┘             └─────────────────┘
```

Use separate async ports for each direction, or connect settings via `property_map`.

### Async Port for UI Communication

UI blocks use the `Async` port flag for rate-decoupled communication:

```cpp
gr::PortOut<float, gr::Async> scalarOut;   // single value
gr::PortIn<float, gr::Async>  scalarIn;    // single value
gr::MsgPortOut                settingsOut; // key-value settings (value type: gr::property_map)
gr::MsgPortIn                 settingsIn;  // key-value settings (value type: gr::property_map)
```

- **Async ports** are evaluated at UI-friendly rates (typically up to 25 Hz)
- **`property_map` outputs** auto-bind to target block settings by key name
- **Primitive outputs** (float, int, string) require explicit wiring

## Threading Model

```
┌───────────────────────────────────────────────────────────────┐
│ Scheduler Thread(s) -- looping through gr::Graph              │
│  ┌─────────┐    ┌────────┐    ┌────────────┐    ┌──────────┐  │
│  │ Source  │───▶│ Filter │───▶│ processing │───▶│ UI Sink  │  │
│  └─────────┘    └────────┘    └────────────┘    └──────────┘  │
│       ▲              ▲                    process[One,Bulk]() │
└───────┼──────────────┼───────────────────────────────┼────────┘
        │              │ (lock-free or                 │
        │              │  mutex-protected buffer)      │
        ▼              ▼                               ▼
┌───────────────────────────────────────────────────────────────┐
│ UI Renderer (toolkit controlled)                              │
│                                                               │
│  for (block : drawableBlocks) {                               │
│      block->draw(config);  // toolkit-specific                │
│  }                                                            │
│                                                               │
│ Loop rate: e.g. ~25 Hz                    draw(property_map{})│
└───────────────────────────────────────────────────────────────┘
```
- **Scheduler** calls `processBulk()`/`processOne()` for data flow
- **Render loop** (external, toolkit-owned) calls `draw()` for presentation

*N.B. the UI rendering is not required to be a separate thread, can be part of a global UI polling loop, 
or even run a similar (or the same) scheduler as the stream-processing scheduler.*

### Thread Safety

Thread safety between `process[Bulk,One]()` and `draw()` is **application design responsibility**.
The framework does not impose synchronization — blocks sharing mutable state between scheduler and render threads must
implement appropriate guards.

> **Note:** Examples in this document use `std::mutex` to clearly mark critical sections.<br>
> Production code typically uses lock-free designs for lower latency.

Common patterns:

- **Lock-free ring buffers** for producer-consumer data flow
- **Atomic flags** for simple state signaling
- **Double buffering** for complex state snapshots

The `config` parameter in `draw(const property_map& config)` follows a similar principle: its contents are a *
*toolkit-specific API contract** that the application must consistently honour.

## More Complete Example

```cpp
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

// Visualization sink — receives data, renders in draw()
template<typename T>
requires std::is_arithmetic_v<T>
struct ChartMonitor : gr::Block<ChartMonitor<T>, gr::Drawable<gr::UICategory::Content, "console">> {
    template<typename U, gr::meta::fixed_string description = "", typename... Args>
    using A = gr::Annotated<U, description, Args...>;

    gr::PortIn<T> in;

    A<gr::Size_t, "history length", gr::Doc<"samples in ring buffer">> n_history  = 1000UZ;
    A<bool, "reset view", gr::Doc<"trigger view reset on next draw">>  reset_view = false;

    GR_MAKE_REFLECTABLE(ChartMonitor, in, n_history, reset_view);

    mutable std::mutex _drawMutex;
    std::vector<T>     _history;

    void settingsChanged(const gr::property_map& /*oldSettings*/, gr::property_map& newSettings) {
        if (newSettings.contains("n_history")) {
            std::lock_guard lock(_drawMutex);
            _history.reserve(n_history);
        }
    }

    constexpr void processOne(T sample) noexcept {
        std::lock_guard lock(_drawMutex);
        _history.push_back(sample);
        if (_history.size() > n_history) {
            _history.erase(_history.begin());
        }
    }

    gr::work::Status draw(const gr::property_map& config = {}) noexcept {
        std::lock_guard lock(_drawMutex);
        if (config.contains("reset_view")) {
            reset_view = true;
        }
        renderChart(_history, reset_view); // toolkit-specific
        reset_view = false;
        return gr::work::Status::OK;
    }
};

// Control block — emits settings from user input
template<typename T>
requires std::is_arithmetic_v<T>
struct ParameterSlider : gr::Block<ParameterSlider<T>,
                                   gr::Drawable<gr::UICategory::Toolbar, "console">> {
    template<typename U, gr::meta::fixed_string description = "", typename... Args>
    using A = gr::Annotated<U, description, Args...>;

    MsgPortOut out;

    A<std::string, "target key", gr::Doc<"settings key to emit">> target_key = "gain";
    A<T, "value">                                                  value     = T{1};
    A<T, "min value">                                              min_value = T{0};
    A<T, "max value">                                              max_value = T{10};

    GR_MAKE_REFLECTABLE(ParameterSlider, out, target_key, value, min_value, max_value);

    gr::work::Status draw(const gr::property_map& /*config*/ = {}) noexcept {
        T new_value = value;
        if (renderSlider(min_value, max_value, new_value)) { // toolkit-specific
            value = new_value;
            out.publish(gr::property_map{{target_key, value}});
        }
        return gr::work::Status::OK;
    }
};

// application setup
void runApplication() {
    gr::Graph graph;

    auto& source = graph.emplaceBlock<SignalSource<float>>({{"sample_rate", 1000.f}});
    auto& filter = graph.emplaceBlock<GainFilter<float>>();
    auto& chart  = graph.emplaceBlock<ChartMonitor<float>>();
    auto& slider = graph.emplaceBlock<ParameterSlider<float>>({{"target_key", "gain"}});

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(source).to<"in">(filter)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(filter).to<"in">(chart)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out">(slider).to<"settings">(filter)));

    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("scheduler init failed: {}", ret.error()));
    }

    // UI thread — toolkit-specific render loop
    std::thread ui_thread([&chart, &slider]() {
        while (true /* find a suitable way to exit the UI thread */) {
            slider.draw();
            std::this_thread::sleep_for(std::chrono::milliseconds(40)); // ~25 Hz
        }
    });

    expect(sched.runAndWait().has_value());
    ui_thread.join();
}
```

## API Reference Summary

### `UICategory` Enum and `Drawable<UICategory, <toolkit>>` NTTP

```cpp
enum class UICategory {
    None,         // No UI contribution (default)
    MenuBar,      // Global app menu bar items
    Toolbar,      // Compact, frequently used actions/toggles
    StatusBar,    // Always-visible, low-interaction status
    Content,      // Primary viewport output
    Panel,        // Secondary panels
    Overlay,      // Layered HUD over Content
    ContextMenu,  // Right-click contextual menus
    Dialog,       // Focused workflows (non-modal)
    Notification  // Transient feedback (toast/banner)
};
```

```cpp
template<UICategory category_, gr::meta::fixed_string toolkit_ = "">
struct Drawable {
    static constexpr UICategory             kCategory = category_;
    static constexpr gr::meta::fixed_string kToolkit  = toolkit_;
};
```

### Block Members (inherited)

| Member             | Type           | Description                           |
|--------------------|----------------|---------------------------------------|
| `ui_constraints`   | `property_map` | Toolkit-specific layout hints         |
| `meta_information` | `property_map` | Contains `"Drawable"` key if drawable |

### BlockModel Interface

| Method         | Returns        | Description                        |
|----------------|----------------|------------------------------------|
| `uiCategory()` | `UICategory`   | Runtime category query             |
| `draw(config)` | `work::Status` | Invoke block's draw implementation |

## See Also

- `gr::testing::ImChartMonitor` — Reference console chart implementation
