#pragma once
#include <vector>
#include <ranges>

namespace solver {

struct SimPoint { double t, y; };

// Downsample a dense solution vector using C++23 std::views::stride (zero-copy).
// Falls back to manual stride on compilers that don't yet ship std::views::stride.
#ifdef __cpp_lib_ranges_stride
inline auto stride_view(const std::vector<double>& solution, std::size_t n) {
    return solution | std::views::stride(n);
}
#else
inline std::vector<double> stride_view(const std::vector<double>& solution, std::size_t n) {
    std::vector<double> out;
    for (std::size_t i = 0; i < solution.size(); i += n)
        out.push_back(solution[i]);
    return out;
}
#endif

// Return SimPoints above a threshold using std::views::filter + enumerate.
// Uses C++23 std::views::enumerate when available, else a manual loop.
#ifdef __cpp_lib_ranges_enumerate
inline auto above_threshold(const std::vector<double>& solution, double dt, double threshold) {
    return solution
        | std::views::enumerate
        | std::views::filter([threshold](auto&& p) { return std::get<1>(p) > threshold; })
        | std::views::transform([dt](auto&& p) -> SimPoint {
              return {static_cast<double>(std::get<0>(p)) * dt, std::get<1>(p)};
          });
}
#else
inline std::vector<SimPoint> above_threshold(const std::vector<double>& solution,
                                              double dt, double threshold) {
    std::vector<SimPoint> out;
    for (std::size_t i = 0; i < solution.size(); ++i)
        if (solution[i] > threshold)
            out.push_back({static_cast<double>(i) * dt, solution[i]});
    return out;
}
#endif

} // namespace solver
