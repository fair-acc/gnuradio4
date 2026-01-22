#include <iostream>
#include <print>
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;
    std::println("device name: {}", q.get_device().get_info<sycl::info::device::name>());
    return 0;
}
