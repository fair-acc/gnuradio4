# Plugins based on the blocklib generator
gr_add_block_plugin(
  GoodBase
  SOURCES
  good_base_plugin.cpp
  HEADERS
  good_base_plugin.hpp)

# Ordinary plugins
add_non_blocklib_plugin(good_math_plugin)
add_non_blocklib_plugin(good_conversion_plugin)
add_non_blocklib_plugin(bad_plugin)
