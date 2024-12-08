## ExprTk GR4's Precompiled Static Library Wrapper 

GR4 provides a wrapper for the [ExprTk library](https://github.com/ArashPartow/exprtk), offering a precompiled static 
library (`libexprtk.a`) to optimise build times. By precompiling templates for `float` and `double`, this wrapper 
avoids redundant template instantiations across multiple targets, significantly improving overall compilation times.


## Why Use This Wrapper?

ExprTk is a versatile toolkit for mathematical expression parsing and evaluation, but its template-heavy design can 
lead to slow builds. Precompiling the most commonly used types (`float` and `double`) into a static library mitigates 
this issue.

### Performance Gains:

1. **Static Library Build Time**: ~1m55s
2. **Dependent Target Build Time**:
    - **With `libexprtk.a`**: ~17–22s
    - **Without `libexprtk.a`**: ~1 minute per target

#### Example Timing Comparisons

| Build Task                | with Wrapper (`libexprtk.a`) | without Wrapper |
|---------------------------|------------------------------|-----------------|
| Static Library (`exprtk`) | ~1m55s                       | -               |
| exprtk_example0           | ~22s                         | ~1m0.5s         |
| exprtk_example1           | ~17.8s                       | ~1m0.2s         |

By using the wrapper, dependent target build times are reduced by ~65%.

## Provided Examples
  
 0. Fibonacci Series: computes Fibonacci series using recursive and iterative approaches via `function_compositor`.
 1. Bubble Sort: implements bubble sort, demonstrating passing and iterating over vectors
 2. Savitzky-Golay Filter: applies a 1st-order smoothing filter to noisy signals using custom weights and multi-dimensional iteration.
 4. 1D Real Discrete Fourier Transform: nomen est omen and a more complex example.

For additional examples and advanced usage, visit the official [ExprTk GitHub repository](https://github.com/ArashPartow/exprtk).

## Usage

- **Link the Static Library**:
  ```cmake
  target_link_libraries(<target> PRIVATE exprtk)
  ```
- **Include the Wrapper**:
  ```cpp
  #include <exprtk.hpp>
  ```

- **ExprTk usage without the Wrapper**: just include
  ```cpp
  #include <exprtk/exprtk.hpp>
  ```

### Acknowledgments
The **ExprTk library** is instrumental in providing efficient mathematical expression parsing and evaluation for GR4. 
It is developed and maintained by **Arash Partow** under the [MIT License](https://github.com/ArashPartow/exprtk/blob/master/license.txt). 
Special thanks to **@ArashPartow** for creating and maintaining such a robust and versatile library.

---