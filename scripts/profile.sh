#!/usr/bin/env bash
# Profile the sphere fitter on 1M points using Valgrind Callgrind.
# Usage: ./scripts/profile.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build_profile"
SCRIPT_DIR="$REPO_ROOT/scripts"
CSV_FILE="$SCRIPT_DIR/points_1m.csv"

echo "=== Building RelWithDebInfo ==="
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF >/dev/null
cmake --build "$BUILD_DIR" --parallel 4 --target sphere_fitter_bench 2>/dev/null || true

if [ ! -f "$BUILD_DIR/sphere_fitter_bench" ]; then
  echo "Building bench executable..."
  cat > /tmp/bench_main.cpp << 'EOF'
#include "PointCloud.h"
#include "SphereFitter.h"
#include <iostream>
int main() {
    auto pts = generate_noisy_sphere({1,2,3}, 5.0, 1000000, 0.01, 42);
    SphereFitter fitter;
    auto result = fitter.fit(pts);
    std::cout << "center=" << result.center.transpose()
              << " radius=" << result.radius
              << " rms=" << result.rms_residual << "\n";
    return 0;
}
EOF
  g++ -std=c++17 -O2 -g -I"$REPO_ROOT/include" \
      -I"$BUILD_DIR/_deps/eigen-src" \
      /tmp/bench_main.cpp \
      "$REPO_ROOT/src/PointCloud.cpp" \
      "$REPO_ROOT/src/SphereFitter.cpp" \
      "$REPO_ROOT/src/RansacFilter.cpp" \
      -o "$BUILD_DIR/sphere_fitter_bench"
fi

echo "=== Running Valgrind Callgrind on 1M points ==="
CALLGRIND_OUT="$SCRIPT_DIR/callgrind.out"
valgrind --tool=callgrind \
         --collect-jumps=yes \
         --callgrind-out-file="$CALLGRIND_OUT" \
         "$BUILD_DIR/sphere_fitter_bench"

echo ""
echo "=== Top 90% functions ==="
callgrind_annotate --threshold=90 "$CALLGRIND_OUT" | head -60

echo ""
echo "Callgrind output written to: $CALLGRIND_OUT"
echo "Open with: kcachegrind $CALLGRIND_OUT"
