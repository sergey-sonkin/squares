use crate::animation::*;
use crate::geometry::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Individual {
    pub squares: Vec<Square>,
    pub container_size: f64,
    pub fitness: f64,
    pub generation: usize,
}

impl Individual {
    pub fn new(squares: Vec<Square>, container_size: f64, generation: usize) -> Self {
        let fitness = Self::calculate_fitness(&squares, container_size);
        Self {
            squares,
            container_size,
            fitness,
            generation,
        }
    }

    fn calculate_fitness(squares: &[Square], container_size: f64) -> f64 {
        let mut fitness = container_size * 1000.0; // Primary objective: minimize container size

        // Heavy penalty for overlaps
        for (i, sq1) in squares.iter().enumerate() {
            for sq2 in squares.iter().skip(i + 1) {
                if sq1.overlaps_with(sq2) {
                    fitness += 50000.0;
                }
            }

            // Penalty for being outside container
            if !sq1.is_inside_container(container_size) {
                fitness += 50000.0;
            }
        }

        // Small bonus for packing efficiency
        let used_area: f64 = squares.len() as f64;
        let total_area = container_size * container_size;
        fitness -= (used_area / total_area) * 100.0;

        fitness
    }

    pub fn is_valid(&self) -> bool {
        // Check no overlaps
        for (i, sq1) in self.squares.iter().enumerate() {
            for sq2 in self.squares.iter().skip(i + 1) {
                if sq1.overlaps_with(sq2) {
                    return false;
                }
            }
            if !sq1.is_inside_container(self.container_size) {
                return false;
            }
        }
        true
    }

    pub fn repair(&mut self, square_size: f64) {
        // Try to fix overlaps and container violations
        let mut rng = thread_rng();

        for _ in 0..100 {
            // Max repair attempts
            let mut fixed = true;

            // Fix container violations
            for square in &mut self.squares {
                if !square.is_inside_container(self.container_size) {
                    // Move square inside container
                    let (min_pt, max_pt) = square.bounding_box();
                    let width = max_pt.x - min_pt.x;
                    let height = max_pt.y - min_pt.y;

                    if width <= self.container_size && height <= self.container_size {
                        square.x = square.x.max(0.0).min(self.container_size - width);
                        square.y = square.y.max(0.0).min(self.container_size - height);
                    } else {
                        // Rotate to fit if possible
                        square.angle = 0.0;
                        square.x = square.x.max(0.0).min(self.container_size - square_size);
                        square.y = square.y.max(0.0).min(self.container_size - square_size);
                    }
                    fixed = false;
                }
            }

            // Fix overlaps by moving squares
            for i in 0..self.squares.len() {
                for j in (i + 1)..self.squares.len() {
                    if self.squares[i].overlaps_with(&self.squares[j]) {
                        // Move one square randomly
                        let move_idx = if rng.gen_bool(0.5) { i } else { j };
                        let square = &mut self.squares[move_idx];

                        let max_move = self.container_size * 0.1;
                        let dx = rng.gen_range(-max_move..max_move);
                        let dy = rng.gen_range(-max_move..max_move);

                        square.x = (square.x + dx)
                            .max(0.0)
                            .min(self.container_size - square_size);
                        square.y = (square.y + dy)
                            .max(0.0)
                            .min(self.container_size - square_size);

                        fixed = false;
                    }
                }
            }

            if fixed {
                break;
            }
        }

        // Recalculate fitness after repair
        self.fitness = Self::calculate_fitness(&self.squares, self.container_size);
    }
}

pub struct GeneticSolver {
    num_squares: usize,
    square_size: f64,
    allow_rotation: bool,
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    elitism_count: usize,
    population: Vec<Individual>,
    generation: usize,
    best_fitness_history: Vec<f64>,
    diversity_history: Vec<f64>,
    animation_recorder: Option<AnimationRecorder>,
}

impl GeneticSolver {
    pub fn new(num_squares: usize, square_size: f64, allow_rotation: bool) -> Self {
        let population_size = (num_squares * 8).max(30).min(150); // Smaller, more achievable population
        let elitism_count = population_size / 10;

        Self {
            num_squares,
            square_size,
            allow_rotation,
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_count,
            population: Vec::new(),
            generation: 0,
            best_fitness_history: Vec::new(),
            diversity_history: Vec::new(),
            animation_recorder: None,
        }
    }

    pub fn with_animation_recording(mut self, frame_interval: usize) -> Self {
        self.animation_recorder = Some(AnimationRecorder::new(
            "Genetic Algorithm".to_string(),
            self.num_squares,
            self.square_size,
            self.allow_rotation,
            frame_interval,
        ));
        self
    }

    pub fn save_animation_data<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref recorder) = self.animation_recorder {
            recorder.save_to_file(path)?;
            println!("Animation data saved");
        } else {
            return Err("No animation data to save".into());
        }
        Ok(())
    }

    pub fn solve(
        &mut self,
        max_generations: usize,
    ) -> Result<(Vec<Square>, f64), Box<dyn std::error::Error>> {
        // Initialize population
        self.initialize_population()?;

        println!("Initial population: {} individuals", self.population.len());
        println!(
            "Best initial fitness: {:.2}",
            self.get_best_individual().fitness
        );

        // Record initial frame
        let best_individual = self.get_best_individual();
        let initial_fitness = best_individual.fitness;
        let initial_container = best_individual.container_size;
        let initial_squares = best_individual.squares.clone();
        let initial_diversity = self.calculate_diversity();

        if let Some(ref mut recorder) = self.animation_recorder {
            let state = AlgorithmState::GeneticAlgorithm {
                generation: 0,
                best_fitness: initial_fitness,
                diversity: initial_diversity,
                population_size: self.population_size,
                mutation_rate: self.mutation_rate,
            };
            recorder.record_frame(
                0,
                &initial_squares,
                initial_container,
                state,
                false,
                Some("Initial population".to_string()),
            );
        }

        for generation in 0..max_generations {
            self.generation = generation;

            // Evolve population
            self.evolve_generation();

            // Track statistics
            let best_individual = self.get_best_individual();
            let best_fitness = best_individual.fitness;
            let best_container = best_individual.container_size;
            let best_squares = best_individual.squares.clone();
            let diversity = self.calculate_diversity();
            let improvement = if let Some(last_fitness) = self.best_fitness_history.last() {
                best_fitness < *last_fitness
            } else {
                false
            };

            self.best_fitness_history.push(best_fitness);
            self.diversity_history.push(diversity);

            // Smart animation recording - only capture meaningful changes
            if let Some(ref mut recorder) = self.animation_recorder {
                let state = AlgorithmState::GeneticAlgorithm {
                    generation: generation + 1,
                    best_fitness,
                    diversity,
                    population_size: self.population_size,
                    mutation_rate: self.mutation_rate,
                };

                let should_record =
                    // Always record first few generations to show initial chaos
                    generation < 5 ||
                    // Always record improvements
                    improvement ||
                    // Record when diversity changes significantly (population evolving)
                    (generation > 0 && (diversity - self.diversity_history[self.diversity_history.len() - 2]).abs() > 0.1) ||
                    // Record periodically but less frequently as we progress
                    (generation < 50 && generation % 3 == 0) ||  // Every 3rd generation early on
                    ((50..100).contains(&generation) && generation % 8 == 0) ||  // Every 8th generation mid-game
                    (generation >= 100 && generation % 15 == 0) ||  // Every 15th generation late game
                    // Record major container size changes
                    (generation > 0 &&
                     recorder.data.frames.last().map_or(true, |last_frame|
                         (last_frame.container_size - best_container).abs() > 0.05));

                if should_record {
                    let event_description = if generation < 5 {
                        Some("Early exploration".to_string())
                    } else if improvement {
                        Some(format!("Improvement: {:.4}", best_fitness))
                    } else if diversity > 0.5 {
                        Some("High diversity - active evolution".to_string())
                    } else if diversity < 0.1 {
                        Some("Low diversity - converging".to_string())
                    } else {
                        None
                    };

                    recorder.record_frame(
                        generation + 1,
                        &best_squares,
                        best_container,
                        state.clone(),
                        improvement,
                        event_description,
                    );
                }

                // Always record significant fitness improvements as special events
                if improvement {
                    recorder.record_significant_event(
                        generation + 1,
                        &best_squares,
                        best_container,
                        state,
                        &format!("New best: {:.4} (gen {})", best_fitness, generation + 1),
                    );
                }
            }

            // Print progress
            if generation % 50 == 0 || generation < 10 {
                println!(
                    "Gen {}: Best fitness: {:.2}, Container: {:.4}, Diversity: {:.3}",
                    generation, best_fitness, best_container, diversity
                );
            }

            // Early termination if no improvement (stricter convergence)
            if generation > 150
                && self.best_fitness_history.len() > 75
                && self
                    .best_fitness_history
                    .iter()
                    .rev()
                    .take(75)
                    .all(|&f| (f - best_fitness).abs() < 0.01)
            {
                println!("Converged at generation {} (strict threshold)", generation);
                break;
            }
        }

        // Finalize animation recording
        if let Some(ref mut recorder) = self.animation_recorder {
            recorder.finalize();
        }

        let best = self.get_best_individual();
        if best.is_valid() {
            Ok((best.squares.clone(), best.container_size))
        } else {
            Err("Best solution is not valid".into())
        }
    }

    fn initialize_population(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.population.clear();

        let theoretical_min = (self.num_squares as f64).sqrt() * self.square_size;
        let container_sizes: Vec<f64> = (0..self.population_size)
            .map(|i| {
                // Create much more diverse initial containers for visual variety
                let ratio = i as f64 / self.population_size as f64;
                theoretical_min * (1.3 + ratio * 1.2) // Range from 1.3x to 2.5x theoretical minimum
            })
            .collect();

        // Generate individuals in parallel
        let individuals: Vec<Individual> = container_sizes
            .par_iter()
            .enumerate()
            .filter_map(|(i, &container_size)| {
                self.generate_random_individual(container_size, i % 4) // Different strategies
            })
            .collect();

        if individuals.is_empty() {
            return Err("Failed to generate any valid initial population".into());
        }

        println!(
            "Generated {} individuals out of {} target",
            individuals.len(),
            self.population_size
        );

        self.population = individuals;
        self.population
            .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        Ok(())
    }

    fn generate_random_individual(
        &self,
        container_size: f64,
        strategy: usize,
    ) -> Option<Individual> {
        let mut rng = thread_rng();
        let mut squares = Vec::new();
        let max_attempts = 2000;

        for _ in 0..self.num_squares {
            let mut placed = false;

            for _ in 0..max_attempts {
                let (x, y, angle) = match strategy {
                    0 => {
                        // Random placement
                        let x = rng.gen::<f64>() * (container_size - self.square_size);
                        let y = rng.gen::<f64>() * (container_size - self.square_size);
                        let angle = if self.allow_rotation {
                            rng.gen::<f64>() * PI / 2.0
                        } else {
                            0.0
                        };
                        (x, y, angle)
                    }
                    1 => {
                        // Edge-biased placement
                        let edge_bias = 0.3;
                        let x = if rng.gen_bool(edge_bias) {
                            if rng.gen_bool(0.5) {
                                0.0
                            } else {
                                container_size - self.square_size
                            }
                        } else {
                            rng.gen::<f64>() * (container_size - self.square_size)
                        };
                        let y = if rng.gen_bool(edge_bias) {
                            if rng.gen_bool(0.5) {
                                0.0
                            } else {
                                container_size - self.square_size
                            }
                        } else {
                            rng.gen::<f64>() * (container_size - self.square_size)
                        };
                        let angle = if self.allow_rotation {
                            rng.gen::<f64>() * PI / 2.0
                        } else {
                            0.0
                        };
                        (x, y, angle)
                    }
                    2 => {
                        // Grid-like placement with noise
                        let grid_size = (self.num_squares as f64).sqrt().ceil() as usize;
                        let cell_size = container_size / grid_size as f64;
                        let grid_x = squares.len() % grid_size;
                        let grid_y = squares.len() / grid_size;

                        let noise = cell_size * 0.3;
                        let x = (grid_x as f64 * cell_size + rng.gen_range(-noise..noise))
                            .max(0.0)
                            .min(container_size - self.square_size);
                        let y = (grid_y as f64 * cell_size + rng.gen_range(-noise..noise))
                            .max(0.0)
                            .min(container_size - self.square_size);
                        let angle = if self.allow_rotation {
                            rng.gen::<f64>() * PI / 2.0
                        } else {
                            0.0
                        };
                        (x, y, angle)
                    }
                    _ => {
                        // Corner-first placement
                        let corners = [
                            (0.0, 0.0),
                            (container_size - self.square_size, 0.0),
                            (0.0, container_size - self.square_size),
                            (
                                container_size - self.square_size,
                                container_size - self.square_size,
                            ),
                        ];

                        if squares.len() < 4 {
                            let (x, y) = corners[squares.len()];
                            let angle = if self.allow_rotation {
                                rng.gen::<f64>() * PI / 2.0
                            } else {
                                0.0
                            };
                            (x, y, angle)
                        } else {
                            let x = rng.gen::<f64>() * (container_size - self.square_size);
                            let y = rng.gen::<f64>() * (container_size - self.square_size);
                            let angle = if self.allow_rotation {
                                rng.gen::<f64>() * PI / 2.0
                            } else {
                                0.0
                            };
                            (x, y, angle)
                        }
                    }
                };

                let square = Square::new(x, y, self.square_size, angle);

                if square.is_inside_container(container_size)
                    && !squares.iter().any(|s| square.overlaps_with(s))
                {
                    squares.push(square);
                    placed = true;
                    break;
                }
            }

            if !placed {
                break;
            }
        }

        if squares.len() == self.num_squares {
            Some(Individual::new(squares, container_size, 0))
        } else {
            None
        }
    }

    fn evolve_generation(&mut self) {
        let mut new_population = Vec::new();

        // Elitism: keep best individuals
        self.population
            .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        for i in 0..self.elitism_count {
            new_population.push(self.population[i].clone());
        }

        // Generate offspring
        while new_population.len() < self.population_size {
            let parent1 = self.tournament_selection();
            let parent2 = self.tournament_selection();

            let mut offspring = if thread_rng().gen::<f64>() < self.crossover_rate {
                self.crossover(&parent1, &parent2)
            } else {
                parent1.clone()
            };

            if thread_rng().gen::<f64>() < self.mutation_rate {
                self.mutate(&mut offspring, self.generation);
            }

            offspring.generation = self.generation;
            offspring.repair(self.square_size);

            new_population.push(offspring);
        }

        self.population = new_population;

        // Adaptive mutation rate based on generation and diversity
        let diversity = self.calculate_diversity();
        let exploration_factor = ((200.0 - self.generation as f64) / 200.0).max(0.1).min(1.0);

        // Start with high mutation rate, decrease over time
        let base_mutation_rate = 0.05 + 0.25 * exploration_factor; // 5% to 30%

        // Adjust based on diversity
        if diversity < 0.1 {
            // Low diversity: increase mutation to explore more
            self.mutation_rate = (base_mutation_rate * 1.5).min(0.4);
        } else if diversity > 0.5 {
            // High diversity: can reduce mutation slightly
            self.mutation_rate = (base_mutation_rate * 0.8).max(0.05);
        } else {
            // Normal diversity: use base rate
            self.mutation_rate = base_mutation_rate;
        }
    }

    fn tournament_selection(&self) -> Individual {
        let tournament_size = 5;
        let mut rng = thread_rng();

        let mut best = &self.population[rng.gen_range(0..self.population.len())];
        for _ in 1..tournament_size {
            let candidate = &self.population[rng.gen_range(0..self.population.len())];
            if candidate.fitness < best.fitness {
                best = candidate;
            }
        }

        best.clone()
    }

    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> Individual {
        let mut rng = thread_rng();

        // Spatial crossover: split container and take squares from each parent
        let split_x = parent1.container_size / 2.0;
        let mut offspring_squares = Vec::new();

        // Take squares from left side of parent1
        for square in &parent1.squares {
            if square.center().x < split_x {
                offspring_squares.push(*square);
            }
        }

        // Take squares from right side of parent2, avoiding overlaps
        for square in &parent2.squares {
            if square.center().x >= split_x {
                let mut new_square = *square;

                // Check for overlaps and adjust if needed
                let mut attempts = 0;
                while attempts < 20
                    && offspring_squares
                        .iter()
                        .any(|s| new_square.overlaps_with(s))
                {
                    new_square.x += rng.gen_range(-0.5..0.5);
                    new_square.y += rng.gen_range(-0.5..0.5);
                    new_square.x = new_square
                        .x
                        .max(0.0)
                        .min(parent1.container_size - self.square_size);
                    new_square.y = new_square
                        .y
                        .max(0.0)
                        .min(parent1.container_size - self.square_size);
                    attempts += 1;
                }

                if !offspring_squares
                    .iter()
                    .any(|s| new_square.overlaps_with(s))
                {
                    offspring_squares.push(new_square);
                }
            }
        }

        // Fill missing squares with random placement
        while offspring_squares.len() < self.num_squares {
            for _ in 0..100 {
                let x = rng.gen::<f64>() * (parent1.container_size - self.square_size);
                let y = rng.gen::<f64>() * (parent1.container_size - self.square_size);
                let angle = if self.allow_rotation {
                    rng.gen::<f64>() * PI / 2.0
                } else {
                    0.0
                };

                let square = Square::new(x, y, self.square_size, angle);

                if square.is_inside_container(parent1.container_size)
                    && !offspring_squares.iter().any(|s| square.overlaps_with(s))
                {
                    offspring_squares.push(square);
                    break;
                }
            }

            if offspring_squares.len() == self.num_squares {
                break;
            }

            // If we can't place more squares, break to avoid infinite loop
            break;
        }

        let container_size = (parent1.container_size + parent2.container_size) / 2.0;
        Individual::new(offspring_squares, container_size, self.generation)
    }

    fn mutate(&self, individual: &mut Individual, generation: usize) {
        let mut rng = thread_rng();

        // Adaptive mutation: start aggressive, become refined
        let max_generations = 200.0; // Assume reasonable max
        let exploration_factor = ((max_generations - generation as f64) / max_generations)
            .max(0.1) // Never go below 10% exploration
            .min(1.0);

        // Choose mutation type with different probabilities
        let mutation_choice = rng.gen::<f64>();

        if mutation_choice < 0.4 {
            // Position mutation (40% chance) - adaptive magnitude
            if !individual.squares.is_empty() {
                let idx = rng.gen_range(0..individual.squares.len());
                let square = &mut individual.squares[idx];

                // Early generations: large moves, later: small refinements
                let base_delta = individual.container_size * (0.05 + 0.4 * exploration_factor);
                let dx = rng.gen_range(-base_delta..base_delta);
                let dy = rng.gen_range(-base_delta..base_delta);

                square.x = (square.x + dx)
                    .max(0.0)
                    .min(individual.container_size - self.square_size);
                square.y = (square.y + dy)
                    .max(0.0)
                    .min(individual.container_size - self.square_size);
            }
        } else if mutation_choice < 0.7 && self.allow_rotation {
            // Angle mutation (30% chance when rotation is enabled) - adaptive rotation
            if !individual.squares.is_empty() {
                let idx = rng.gen_range(0..individual.squares.len());
                let square = &mut individual.squares[idx];

                // Early generations: dramatic rotations, later: fine adjustments
                let dramatic_rotation_chance = 0.1 + 0.6 * exploration_factor; // 10-70% chance

                if rng.gen::<f64>() < dramatic_rotation_chance {
                    // Complete rotation change (more likely early on)
                    square.angle = rng.gen::<f64>() * PI / 2.0;
                } else {
                    // Incremental adjustment (magnitude decreases over time)
                    let max_angle_delta = (PI / 12.0) + (PI / 3.0) * exploration_factor; // π/12 to π/3
                    let angle_delta = rng.gen_range(-max_angle_delta..max_angle_delta);
                    square.angle = (square.angle + angle_delta) % (PI / 2.0);
                }
            }
        } else if mutation_choice < 0.85 {
            // Swap mutation (15% chance) - more likely early on
            if individual.squares.len() >= 2 && rng.gen::<f64>() < (0.5 + 0.5 * exploration_factor)
            {
                let idx1 = rng.gen_range(0..individual.squares.len());
                let idx2 = rng.gen_range(0..individual.squares.len());

                if idx1 != idx2 {
                    // Early generations: swap positions and angles, later: just positions
                    if exploration_factor > 0.5 && self.allow_rotation {
                        // Swap everything
                        let square1 = individual.squares[idx1];
                        let square2 = individual.squares[idx2];

                        individual.squares[idx1].x = square2.x;
                        individual.squares[idx1].y = square2.y;
                        individual.squares[idx1].angle = square2.angle;

                        individual.squares[idx2].x = square1.x;
                        individual.squares[idx2].y = square1.y;
                        individual.squares[idx2].angle = square1.angle;
                    } else {
                        // Just swap positions
                        let pos1 = (individual.squares[idx1].x, individual.squares[idx1].y);
                        let pos2 = (individual.squares[idx2].x, individual.squares[idx2].y);

                        individual.squares[idx1].x = pos2.0;
                        individual.squares[idx1].y = pos2.1;
                        individual.squares[idx2].x = pos1.0;
                        individual.squares[idx2].y = pos1.1;
                    }
                }
            }
        } else {
            // Container size mutation (15% chance) - adaptive magnitude
            let base_delta = 0.02 + 0.08 * exploration_factor; // 2-10% changes
            let delta = rng.gen_range(-base_delta..base_delta);
            individual.container_size = (individual.container_size * (1.0 + delta))
                .max(self.square_size * (self.num_squares as f64).sqrt());
        }

        // Recalculate fitness
        individual.fitness =
            Individual::calculate_fitness(&individual.squares, individual.container_size);
    }

    fn calculate_diversity(&self) -> f64 {
        if self.population.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut comparisons = 0;

        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let distance = self.solution_distance(&self.population[i], &self.population[j]);
                total_distance += distance;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_distance / comparisons as f64
        } else {
            0.0
        }
    }

    fn solution_distance(&self, ind1: &Individual, ind2: &Individual) -> f64 {
        // Simple distance based on container size and average position differences
        let container_diff = (ind1.container_size - ind2.container_size).abs();

        let mut position_diff = 0.0;
        let min_len = ind1.squares.len().min(ind2.squares.len());

        for i in 0..min_len {
            let dx = ind1.squares[i].x - ind2.squares[i].x;
            let dy = ind1.squares[i].y - ind2.squares[i].y;
            position_diff += (dx * dx + dy * dy).sqrt();
        }

        container_diff + position_diff / min_len as f64
    }

    pub fn get_best_individual(&self) -> &Individual {
        self.population
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
    }

    pub fn print_statistics(&self) {
        if self.best_fitness_history.is_empty() {
            return;
        }

        let best = self.get_best_individual();
        let diversity = self.calculate_diversity();

        println!("\nGenetic Algorithm Statistics:");
        println!("Generations: {}", self.generation);
        println!("Population size: {}", self.population_size);
        println!("Best fitness: {:.2}", best.fitness);
        println!("Best container size: {:.6}", best.container_size);
        println!("Final diversity: {:.3}", diversity);
        println!("Final mutation rate: {:.3}", self.mutation_rate);

        let valid_count = self.population.iter().filter(|ind| ind.is_valid()).count();
        println!(
            "Valid solutions in population: {}/{}",
            valid_count,
            self.population.len()
        );
    }
}
