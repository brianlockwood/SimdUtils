#pragma once

#include <utility>
#include <array>

#include <cmath>

#ifdef SSE2
#include <xmmintrin.h>
#endif

#ifdef AVX
#include <immintrin.h>
#endif

template <int Size, typename Real>
struct SimdTypeTraits;


#ifdef SSE2
template <>
struct SimdTypeTraits<2, double>
{
    using Type = __m128d;

    static Type create(double a)
    {
        return _mm_set_pd(a,a);
    }

    static Type create(double a, double b)
    {
        return _mm_set_pd(b,a);
    }

    static Type create(const double *a)
    {
        return _mm_load_pd(a);
    }

    static Type mul(const Type &a, const Type &b)
    {
        return _mm_mul_pd(a, b);
    }

    static Type inv(const Type &a)
    {
        Type one = _mm_set_pd(1.0, 1.0);
        return _mm_div_pd(one, a);
    }

    static Type sub(const Type &a, const Type &b)
    {
        return _mm_sub_pd(a, b);
    }

    static Type add(const Type &a, const Type &b)
    {
        return _mm_add_pd(a, b);
    }

    static void store(std::array<double,2> &result, const Type &a)
    {
        _mm_store_pd(result.data(), a);
    }

    static double reduceAdd(const Type &a)
    {
        std::array<double,2> result;
        _mm_store_pd(result.data(), a);
        return result[0] + result[1];
    }

    static Type cos(const Type &a)
    {
        std::array<double,2> result;
        _mm_store_pd(result.data(), a);
        return _mm_set_pd(std::cos(result[0]), std::cos(result[1]));
    }
};
#elif AVX
template <>
struct SimdTypeTraits<4, double>
{
    using Type = __m256d;

    static Type create(double a)
    {
        return _mm256_set_pd(a,a,a,a);
    }

    static Type create(double a, double b, double c, double d)
    {
        return _mm256_set_pd(d,c,b,a);
    }
    
    static Type create(const double *a)
    {
        return _mm256_load_pd(a);
    }

    static Type mul(const Type &a, const Type &b)
    {
        return _mm256_mul_pd(a, b);
    }

    static Type inv(const Type &a)
    {
        Type one = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
        return _mm256_div_pd(one, a);
    }

    static Type sub(const Type &a, const Type &b)
    {
        return _mm256_sub_pd(a, b);
    }

    static Type add(const Type &a, const Type &b)
    {
        return _mm256_add_pd(a, b);
    }

    static void store(std::array<double,4> &result, const Type &a)
    {
        _mm256_store_pd(result.data(), a);
    }

    static double reduceAdd(const Type &a)
    {
        std::array<double,4> result;
        _mm256_store_pd(result.data(), a);
        return result[0] + result[1] + result[2] + result[3];
    }

    static Type cos(const Type &a)
    {
        std::array<double,4> result;
        _mm256_store_pd(result.data(), a);
        return _mm256_set_pd( std::cos(result[0]), std::cos(result[1]), 
                 std::cos(result[2]), std::cos(result[3]) );
    }
};
#endif

template <int Size, typename Real>
class Scalar
{
public:
    using SimdTraits = SimdTypeTraits<Size, Real>;
    using Simd_t = typename SimdTraits::Type;

    static constexpr int Size2() { return Size; }

    template <typename... Args>
    Scalar(Args... args)
        : data(SimdTraits::create(std::forward<Args>(args)...))
    {}

    Scalar(const Simd_t &data)
        : data(data)
    {}

    Scalar(Simd_t &&data)
        : data(data)
    {}

    Scalar& operator=(const Simd_t &other) 
    {
        data = other;
        return *this;
    }

    Scalar& operator=(Simd_t &&other) 
    {
        data = other;
        return *this;
    }

    // Rule of 5
    Scalar(const Scalar &other)
        : data(other.data)
    {}

    Scalar(Scalar &&other)
        : data(other.data)
    {}

    ~Scalar() = default;

    Scalar& operator=(const Scalar &other) 
    {
        data = other.data;
        return *this;
    }

    Scalar& operator=(Scalar &&other) 
    {
        data = other.data;
        return *this;
    }

    std::array<Real, Size> asArray()
    {
        std::array<Real, Size> result;
        SimdTypeTraits<Size, Real>::store(result, data);

        return result;
    }

    Real reduceAdd()
    {
        return SimdTypeTraits<Size, Real>::reduceAdd(data);
    }

    Simd_t data;
};

template <int Size, typename Real>
inline Scalar<Size, Real> operator*(const Scalar<Size, Real> &a, const Scalar<Size, Real> &b)
{
    return SimdTypeTraits<Size, Real>::mul(a.data,b.data);
}

template <int Size, typename Real>
inline Scalar<Size, Real> operator-(const Scalar<Size, Real> &a, const Scalar<Size, Real> &b)
{
    return SimdTypeTraits<Size, Real>::sub(a.data,b.data);
}

template <int Size, typename Real>
inline Scalar<Size, Real> operator+(const Scalar<Size, Real> &a, const Scalar<Size, Real> &b)
{
    return SimdTypeTraits<Size, Real>::add(a.data,b.data);
}

template <int Size, typename Real>
inline Scalar<Size, Real> inv(const Scalar<Size, Real> &a)
{
    return SimdTypeTraits<Size, Real>::inv(a.data);
}

inline double inv(double a)
{
    return 1.0/a;
}

#ifdef SSE2
    using ScalarSIMD = Scalar<2,double>; 
#endif

#ifdef AVX
    using ScalarSIMD = Scalar<4,double>; 
#endif

namespace Math
{
template <int Size, typename Real>
inline Scalar<Size, Real> cos(const Scalar<Size, Real> &a)
{
    return SimdTypeTraits<Size, Real>::cos(a.data);
}
inline double cos(double a)
{
    return std::cos(a);
}
}
