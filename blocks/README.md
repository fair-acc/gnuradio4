# gnuradio blocks

This directory contains collections of blocks by topic.

Structure of a block, entries in square brackets are optional:

- <library_name>
  - include/gnuradio-4.0/<library_name>/
    - one or more headers which each can contain one or more block definitions
  - test
    - qa_<blockname> - tests for a block
    - CMakeLists.txt 
  - README.md - a short description of the block library
  - [src] - containing optional samples
  - [assets] - additional block documentation