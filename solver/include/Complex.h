#pragma once
#include "Concepts.h"
#include "MathCore.h"
#include <numbers>
#include <cmath>

namespace solver {

// Constexpr complex number type — all arithmetic evaluable at compile time.
// Satisfies the FloatingPoint concept so it plugs into the solver templates.
template<FloatingPoint T>
struct Complex {
    T re{}, im{};

    constexpr Complex operator+(const Complex& o) const { return {re+o.re, im+o.im}; }
    constexpr Complex operator-(const Complex& o) const { return {re-o.re, im-o.im}; }
    constexpr Complex operator*(const Complex& o) const {
        return {re*o.re - im*o.im, re*o.im + im*o.re};
    }
    constexpr Complex operator*(T scalar) const { return {re*scalar, im*scalar}; }
    constexpr T       abs()              const { return std::sqrt(re*re + im*im); }
    constexpr bool    operator==(const Complex&) const = default;
};

// Compile-time complex exponential: e^z = e^(re) · (cos(im) + i·sin(im))
template<FloatingPoint T>
consteval Complex<T> exp_complex(Complex<T> z) {
    T mag  = taylor_exp<20>(z.re);
    T cosv = taylor_cos<15>(z.im);
    T sinv = taylor_sin<15>(z.im);
    return {mag * cosv, mag * sinv};
}

// Euler's formula: e^(iπ) + 1 ≈ 0  — real part ≈ -1, imag ≈ 0
static_assert(cabs(exp_complex(Complex<double>{0.0, std::numbers::pi}).re + 1.0) < 1e-10);
static_assert(cabs(exp_complex(Complex<double>{0.0, std::numbers::pi}).im)       < 1e-10);

} // namespace solver
