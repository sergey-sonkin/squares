# Square Packing Solver

High-performance solver for packing unit squares into minimum containers using simulated annealing.

## Usage

```bash
# Solve 17 squares problem with visualization
cargo run -- solve --num-squares 17 --rotation --visualize

# Test against known optimal solutions
cargo run -- test --max-squares 10

# Benchmark performance
cargo run -- benchmark --num-squares 17 --runs 5
```

## Results

- 17 squares: ~2.5s, finds solutions 1.35x Bidwell's 1997 optimal (4.675)
- Smaller cases: typically 20-50% above optimal
- Supports square rotation for better packing efficiency

## Implementation

- Simulated annealing with multiple mutation strategies
- SAT collision detection for rotated squares
- Real-time convergence tracking and visualization