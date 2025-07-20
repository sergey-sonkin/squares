# Square Packing Solver

## Implementation

- SAT collision detection for rotated squares
- Real-time convergence tracking and visualization
- Parallel population evaluation
- Comprehensive benchmarking and comparison tools

## Algorithms

### Simulated Annealing
Single-solution optimization that gradually "cools" to avoid local optima:
- Fast convergence (~2-3 seconds)
- Simple implementation
- Good for quick approximations

### Genetic Algorithm
Population-based evolution with superior solution quality:
- Multiple initialization strategies
- Tournament selection and spatial crossover
- Adaptive mutation rates
- Automatic diversity maintenance
- 21% better solutions than simulated annealing

## Usage

### Simulated Annealing
```bash
# Solve 17 squares problem with visualization
cargo run -- solve --num-squares 17 --rotation --visualize

# Quick solve without rotation
cargo run -- solve --num-squares 10 --iterations 10000
```

### Genetic Algorithm
```bash
# Solve using genetic algorithm (better quality)
cargo run -- genetic --num-squares 17 --rotation --visualize

# Faster genetic solve with fewer generations
cargo run -- genetic --num-squares 10 --generations 100 --rotation
```

### Testing & Comparison
```bash
# Test against known optimal solutions
cargo run -- test --max-squares 10

# Compare both algorithms head-to-head
cargo run -- compare --num-squares 17 --runs 3 --rotation

# Benchmark simulated annealing performance
cargo run -- benchmark --num-squares 17 --runs 5
```

## Results

| Algorithm | 17 Squares | Time |
|-----------|------------|------|
| Simulated Annealing | 1.35x optimal | ~2.5s |
| **Genetic Algorithm** | **1.06x optimal** | ~15s |

- Bidwell's 1997 optimal: 4.675
- Genetic algorithm finds solutions only 6% above optimal
- Supports square rotation for better packing efficiency
