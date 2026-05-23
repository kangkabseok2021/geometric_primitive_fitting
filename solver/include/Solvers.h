#pragma once
#include "SolverBase.h"

namespace solver {

// Euler method: O(h) global error, one function evaluation per step.
class EulerSolver : public SolverBase<EulerSolver, double> {
public:
    template<typename F>
    double step(F&& f, double t, double y, double dt) const {
        return y + dt * f(t, y);
    }
};

// Classical Runge-Kutta 4: O(h⁴) global error, four function evaluations per step.
class RK4Solver : public SolverBase<RK4Solver, double> {
public:
    template<typename F>
    double step(F&& f, double t, double y, double dt) const {
        const double k1 = f(t,          y);
        const double k2 = f(t + dt/2.0, y + dt*k1/2.0);
        const double k3 = f(t + dt/2.0, y + dt*k2/2.0);
        const double k4 = f(t + dt,     y + dt*k3);
        return y + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
    }
};

// Compile-time proof: neither solver uses virtual dispatch.
static_assert(!std::is_polymorphic_v<EulerSolver>);
static_assert(!std::is_polymorphic_v<RK4Solver>);

} // namespace solver
