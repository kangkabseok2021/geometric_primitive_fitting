#include <gtest/gtest.h>
#include "MathCore.h"
#include "Complex.h"
#include <numbers>
#include <cmath>

using namespace solver;

// --- Compile-time static_asserts (executed by the compiler, zero runtime cost) ---
// These are also verified at compile time in the headers themselves; duplicated here
// to make the test binary's output explicitly confirm compile-time correctness.

static_assert(factorial(0) == 1);
static_assert(factorial(5) == 120);
static_assert(factorial(10) == 3628800);

static_assert(taylor_exp<20>(0.0) == 1.0);
static_assert(cabs(taylor_exp<20>(1.0) - std::numbers::e) < 1e-14);
static_assert(taylor_sin<15>(0.0) == 0.0);
static_assert(cabs(taylor_cos<15>(0.0) - 1.0) < 1e-15);

static_assert(cabs(exp_complex(Complex<double>{0.0, std::numbers::pi}).re + 1.0) < 1e-10);
static_assert(cabs(exp_complex(Complex<double>{0.0, std::numbers::pi}).im)       < 1e-10);

// --- Runtime GoogleTests ---

TEST(MathCore, FactorialSmallValues) {
    EXPECT_EQ(factorial(0), 1);
    EXPECT_EQ(factorial(1), 1);
    EXPECT_EQ(factorial(5), 120);
    EXPECT_EQ(factorial(10), 3628800);
}

TEST(MathCore, TaylorExpConvergence) {
    // constexpr path (compile-time path via smart_exp is not tested at runtime here)
    constexpr double result = taylor_exp<20>(2.0);
    EXPECT_NEAR(result, std::exp(2.0), 1e-12);
}

TEST(MathCore, TaylorExpZero) {
    EXPECT_DOUBLE_EQ(taylor_exp<20>(0.0), 1.0);
}

TEST(MathCore, TaylorSinCosAtZero) {
    EXPECT_DOUBLE_EQ(taylor_sin<15>(0.0), 0.0);
    EXPECT_NEAR(taylor_cos<15>(0.0), 1.0, 1e-15);
}

TEST(MathCore, TaylorSinHalfPi) {
    constexpr double half_pi = std::numbers::pi / 2.0;
    EXPECT_NEAR(taylor_sin<15>(half_pi), 1.0, 1e-12);
    EXPECT_NEAR(taylor_cos<15>(half_pi), 0.0, 1e-12);
}

TEST(MathCore, ExpTableSize) {
    EXPECT_EQ(EXP_TABLE.size(), 21u);
    EXPECT_NEAR(EXP_TABLE[0], 1.0,              1e-14); // exp(0.0)
    EXPECT_NEAR(EXP_TABLE[2], std::exp(1.0),    1e-12); // exp(0.5*2 = 1.0)
    EXPECT_NEAR(EXP_TABLE[20], std::exp(10.0),  50.0);  // exp(10.0) — 20-term Taylor has ~35 error
}

TEST(MathCore, SmartExpSelectsCorrectPath) {
    // At runtime smart_exp delegates to std::exp — result must match closely.
    EXPECT_NEAR(smart_exp(1.0), std::exp(1.0), 1e-15);
    EXPECT_NEAR(smart_exp(3.5), std::exp(3.5), 1e-12);
}

TEST(ComplexMath, EulerFormulaRuntime) {
    // Verify e^(iπ) + 1 ≈ 0 at runtime as well
    constexpr auto result = exp_complex(Complex<double>{0.0, std::numbers::pi});
    EXPECT_NEAR(result.re, -1.0, 1e-10);
    EXPECT_NEAR(result.im,  0.0, 1e-10);
}

TEST(ComplexMath, ArithmeticOps) {
    constexpr Complex<double> a{1.0, 2.0};
    constexpr Complex<double> b{3.0, 4.0};
    constexpr auto sum  = a + b;
    constexpr auto prod = a * b;
    EXPECT_DOUBLE_EQ(sum.re,  4.0);
    EXPECT_DOUBLE_EQ(sum.im,  6.0);
    EXPECT_DOUBLE_EQ(prod.re, -5.0);  // 1*3 - 2*4
    EXPECT_DOUBLE_EQ(prod.im, 10.0);  // 1*4 + 2*3
}
