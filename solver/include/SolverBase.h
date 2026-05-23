#pragma once
#include "Concepts.h"
#include <vector>

namespace solver {

// CRTP base — solve() dispatches to Derived::step() at compile time with zero vtable overhead.
template<typename Derived, Integrable T>
class SolverBase {
public:
    // Integrate dy/dt = f(t, y) from t0 to t1 with step dt, returning all y values.
    template<typename F>
    [[nodiscard]] std::vector<T> solve(F&& f, T t0, T t1, T y0, T dt) const {
        std::vector<T> result;
        result.reserve(static_cast<std::size_t>((t1 - t0) / dt) + 2);
        T t = t0, y = y0;
        while (t < t1) {
            result.push_back(y);
            y = static_cast<const Derived*>(this)->step(std::forward<F>(f), t, y, dt);
            t += dt;
        }
        return result;
    }
};

} // namespace solver
