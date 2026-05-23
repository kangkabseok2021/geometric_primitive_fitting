#include <gtest/gtest.h>
#include "Solvers.h"
#include "SimulationViews.h"
#include <cmath>
#include <numbers>
#include <type_traits>

using namespace solver;

// Exact analytical solution for dy/dt = -y, y(0) = 1 is y(t) = e^{-t}.
static double exact(double t) { return std::exp(-t); }

static double rms(const std::vector<double>& sol, double dt) {
    double sse = 0.0;
    for (std::size_t i = 0; i < sol.size(); ++i) {
        double err = sol[i] - exact(static_cast<double>(i) * dt);
        sse += err * err;
    }
    return std::sqrt(sse / static_cast<double>(sol.size()));
}

// --- No virtual dispatch ---
TEST(CRTP, EulerSolverIsNotPolymorphic) {
    EXPECT_FALSE(std::is_polymorphic_v<EulerSolver>);
}
TEST(CRTP, RK4SolverIsNotPolymorphic) {
    EXPECT_FALSE(std::is_polymorphic_v<RK4Solver>);
}

// --- Euler: O(h) accuracy ---
TEST(EulerSolver, DecayEquationRMS) {
    EulerSolver s;
    auto sol = s.solve([](double /*t*/, double y) { return -y; }, 0.0, 5.0, 1.0, 1e-3);
    EXPECT_LT(rms(sol, 1e-3), 5e-4);  // O(h) bound
}

TEST(EulerSolver, SolutionNotEmpty) {
    EulerSolver s;
    auto sol = s.solve([](double, double y) { return y; }, 0.0, 1.0, 1.0, 0.1);
    EXPECT_GT(sol.size(), 0u);
}

// --- RK4: O(h⁴) accuracy ---
TEST(RK4Solver, DecayEquationRMS) {
    RK4Solver s;
    // dt = 0.01 → RK4 error O((0.01)⁴) ≈ 1e-8
    auto sol = s.solve([](double /*t*/, double y) { return -y; }, 0.0, 5.0, 1.0, 1e-2);
    EXPECT_LT(rms(sol, 1e-2), 1e-6);
}

TEST(RK4Solver, MoreAccurateThanEulerAtSameStep) {
    auto f = [](double, double y) { return -y; };
    EulerSolver eu;
    RK4Solver   rk;
    double dt = 0.05;
    auto sol_eu = eu.solve(f, 0.0, 2.0, 1.0, dt);
    auto sol_rk = rk.solve(f, 0.0, 2.0, 1.0, dt);
    EXPECT_LT(rms(sol_rk, dt), rms(sol_eu, dt));
}

TEST(RK4Solver, MatchesExactSinusoid) {
    // dy/dt = cos(t) → y(t) = sin(t) + 1
    RK4Solver s;
    auto sol = s.solve([](double t, double /*y*/) { return std::cos(t); },
                       0.0, std::numbers::pi, 0.0, 1e-3);
    // y(π) = sin(π) = 0 — check last value
    EXPECT_NEAR(sol.back(), std::sin(static_cast<double>(sol.size()-1) * 1e-3), 1e-5);
}

TEST(RK4Solver, NodiscardEnforced) {
    // [[nodiscard]] on solve() — verified by compiler warning; runtime: just check it returns
    RK4Solver s;
    auto result = s.solve([](double, double y) { return -y; }, 0.0, 1.0, 1.0, 0.1);
    EXPECT_FALSE(result.empty());
}

// --- SimulationViews ---
TEST(SimulationViews, StrideReducesSize) {
    RK4Solver s;
    auto sol = s.solve([](double, double y) { return -y; }, 0.0, 1.0, 1.0, 1e-3);
    auto strided = stride_view(sol, 10);
    // strided should have ~sol.size()/10 elements
    std::size_t count = 0;
#ifdef __cpp_lib_ranges_stride
    for ([[maybe_unused]] auto v : strided) ++count;
#else
    count = strided.size();
#endif
    EXPECT_NEAR(static_cast<double>(count),
                static_cast<double>(sol.size()) / 10.0, 2.0);
}

TEST(SimulationViews, AboveThresholdFilters) {
    std::vector<double> sol = {0.5, 1.5, 0.8, 2.0, 0.3};
    auto pts = above_threshold(sol, 0.1, 1.0);
    std::size_t count = 0;
#ifdef __cpp_lib_ranges_enumerate
    for ([[maybe_unused]] auto p : pts) ++count;
#else
    count = pts.size();
#endif
    EXPECT_EQ(count, 2u);  // values 1.5 and 2.0
}
