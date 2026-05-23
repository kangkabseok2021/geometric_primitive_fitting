#pragma once
#include <concepts>
#include <type_traits>
#include <cmath>
#include <limits>

namespace solver {

// Admits float, double, long double, and any user-defined numeric type
// that provides +, *, /, and an abs()-convertible magnitude.
template<typename T>
concept FloatingPoint =
    std::is_floating_point_v<T> ||
    requires(T a, T b) {
        { a + b } -> std::same_as<T>;
        { a * b } -> std::same_as<T>;
        { a / b } -> std::same_as<T>;
        { std::abs(a) } -> std::convertible_to<double>;
    };

// Adds epsilon requirement — needed for adaptive step-size control.
template<typename T>
concept Integrable =
    FloatingPoint<T> &&
    requires {
        { std::numeric_limits<T>::epsilon() } -> std::convertible_to<double>;
    };

} // namespace solver
