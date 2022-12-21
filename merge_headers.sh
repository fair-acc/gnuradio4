#! /usr/env bash
mkdir -p dist
# merge headers by precompiling toplevel headers to get list of included headers, filter for only lines with local headers, reverse the list to get deepest deps first
# then concat all headers and filter out local includes and write to single header file
cat $(gcc -std=c++20 -H include/graph.hpp 2>&1 | grep "^[\.][\.]* [^/]" | sed -e "s%^[\.]* %%" | tac ) include/graph.hpp | grep -v "#include \"" > dist/graph.hpp
rm include/graph.hpp.gch # remove generated precompiled header

