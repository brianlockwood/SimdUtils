add_executable(typesBenchmark)
target_sources(typesBenchmark PRIVATE ScalarBench.cpp)
target_link_libraries(typesBenchmark PRIVATE types ${CONAN_LIBS})

