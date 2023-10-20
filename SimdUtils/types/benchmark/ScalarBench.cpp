#include "types/Scalar.h"

#include <benchmark/benchmark.h>

template <typename Real>
Real function(Real a, Real b)
{
  Real c = a + b;
  Real d = c*b;
  Real e = d + c - a*b;
  Real f = inv(a)*e;

  return f;
}


int N = 1000;

static void scalar_add(benchmark::State& state) {

  std::vector<double> a(N);
  std::vector<double> b(N);
  
  for (int i=0; i < N; ++i)
  {
    a[i] = std::rand();
    b[i] = std::rand();
  }

  for (auto _ : state)
  {
    double sum = 0.0;
    for (int i=0; i < N; ++i)
      sum += function(a[i],b[i]);

    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(scalar_add);

static void scalarSimd_add(benchmark::State& state) {

  double *a = new(std::align_val_t{alignof(ScalarSIMD)}) double[N];
  double *b = new(std::align_val_t{alignof(ScalarSIMD)}) double[N];

  for (int i=0; i < N; ++i)
  {
    a[i] = std::rand();
    b[i] = std::rand();
  }

  for (auto _ : state)
  {
    ScalarSIMD sum = 0.0;
    for (int i=0; i < N; i = i + ScalarSIMD::Size2())
      sum = sum + function(ScalarSIMD(&a[i]),
                      ScalarSIMD(&b[i]));

    benchmark::DoNotOptimize(sum.reduceAdd());
  }
}
BENCHMARK(scalarSimd_add);

BENCHMARK_MAIN();