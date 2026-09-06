add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/highway-build/googletest-src" "${CMAKE_CURRENT_BINARY_DIR}/attention-ab/gtest")
add_library(GTest::Main ALIAS gtest_main)
