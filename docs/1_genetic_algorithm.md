# Genetic Algorithm vs Simulated Annealing

Current (Simulated Annealing):
- Single solution that evolves over time
- Uses temperature-based acceptance of worse solutions
- Gradual cooling reduces exploration

Genetic Algorithm:
- Population of solutions that evolve together
- Solutions "mate" to create offspring
- Natural selection keeps best solutions

How Genetic Algorithm Works for Square Packing

1. Population Representation

Each individual = one complete square packing solution
```
```
struct Individual {
    squares: Vec<Square>,
    container_size: f64,
    fitness: f64,  // Lower = better (inverse of container size)
}
```
```
2. Core Operations

Selection: Pick parents based on fitness (tournament, roulette wheel)

Crossover: Combine two parent solutions
- Take positions from parent A, angles from parent B
- Spatial crossover: split container in half, take squares from each parent
- Order crossover: sequence of square placements

Mutation: Random changes (same as our current mutations)
- Position perturbation
- Angle adjustment
- Square swapping

3. Evolution Loop

1. Generate initial population (random packings)
2. Evaluate fitness (container size needed)
3. Select parents for reproduction
4. Create offspring via crossover + mutation
5. Replace worst individuals with offspring
6. Repeat until convergence

Implementation Plan

Phase 1: Core GA Structure

- Population struct managing multiple individuals
- Selection algorithms (tournament selection)
- Basic crossover operators

Phase 2: Domain-Specific Operators

- Spatial crossover for square positions
- Repair mechanisms for invalid offspring
- Adaptive mutation rates

Phase 3: Hybrid Approach

- Use GA for global exploration
- Apply local search (simulated annealing) to best individuals
- Multi-objective optimization (minimize container + maximize packing density)

Advantages vs Current Approach

- Diversity: Multiple solutions prevent local optima
- Parallelization: Evaluate population in parallel
- Robustness: Less sensitive to initial conditions

Challenges

- Representation: Ensuring offspring are valid packings
- Premature convergence: Population becoming too similar
- Parameter tuning: Population size, crossover/mutation rates
