#pragma once
#include "Concepts.h"
#include <array>
#include <numbers>
#include <cmath>

namespace solver {

// constexpr absolute value — std::abs is not constexpr on all C++23 compilers yet.
template<typename T>
constexpr T cabs(T x) { return x < T{0} ? -x : x; }

// Compile-time factorial — negative n is a compile-time hard error.
consteval int factorial(int n) {
    if (n < 0) throw std::domain_error("negative factorial");
    return n == 0 ? 1 : n * factorial(n - 1);
}

// N-term Taylor exp: Σₖ₌₀ᴺ xᵏ/k!
template<int N, FloatingPoint T>
constexpr T taylor_exp(T x) {
    T sum = T{1}, term = T{1};
    for (int k = 1; k <= N; ++k) {
        term = term * (x / T(k));
        sum  = sum  + term;
    }
    return sum;
}

// N-term Taylor sin: Σₖ₌₀ᴺ (-1)ᵏ x^(2k+1)/(2k+1)!
template<int N, FloatingPoint T>
constexpr T taylor_sin(T x) {
    T sum = x, term = x;
    for (int k = 1; k <= N; ++k) {
        term = -term * (x * x) / T((2*k) * (2*k + 1));
        sum  = sum + term;
    }
    return sum;
}

// N-term Taylor cos: Σₖ₌₀ᴺ (-1)ᵏ x^(2k)/(2k)!
template<int N, FloatingPoint T>
constexpr T taylor_cos(T x) {
    T sum = T{1}, term = T{1};
    for (int k = 1; k <= N; ++k) {
        term = -term * (x * x) / T((2*k - 1) * (2*k));
        sum  = sum + term;
    }
    return sum;
}

// C++23 if consteval: compile-time → Taylor<20>; runtime → std::exp (hardware FPU).
template<int N = 20, FloatingPoint T>
constexpr T smart_exp(T x) {
    if consteval {
        return taylor_exp<N>(x);
    } else {
        return std::exp(x);
    }
}

// Compile-time lookup table: exp(0.0), exp(0.5), ..., exp(10.0) — 21 entries.
// Materialised entirely by the compiler; zero runtime instructions for table access.
inline constexpr auto EXP_TABLE = []() constexpr {
    std::array<double, 21> t{};
    for (int i = 0; i <= 20; ++i)
        t[static_cast<std::size_t>(i)] = taylor_exp<20>(double(i) * 0.5);
    return t;
}();

// Compile-time correctness proofs — verified at zero runtime cost.
static_assert(taylor_exp<20>(0.0) == 1.0);
static_assert(taylor_sin<15>(0.0) == 0.0);
static_assert(cabs(taylor_cos<15>(0.0) - 1.0) < 1e-15);
static_assert(cabs(taylor_exp<20>(1.0) - std::numbers::e) < 1e-14);

} // namespace solver
