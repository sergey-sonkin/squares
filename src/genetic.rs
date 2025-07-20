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
        
        for _ in 0..100 { // Max repair attempts
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
                        
                        square.x = (square.x + dx).max(0.0).min(self.container_size - square_size);
                        square.y = (square.y + dy).max(0.0).min(self.container_size - square_size);
                        
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
}

impl GeneticSolver {
    pub fn new(num_squares: usize, square_size: f64, allow_rotation: bool) -> Self {
        let population_size = (num_squares * 10).max(50).min(200);
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
        }
    }
    
    pub fn solve(&mut self, max_generations: usize) -> Result<(Vec<Square>, f64), Box<dyn std::error::Error>> {
        // Initialize population
        self.initialize_population()?;
        
        println!("Initial population: {} individuals", self.population.len());
        println!("Best initial fitness: {:.2}", self.get_best_individual().fitness);
        
        for generation in 0..max_generations {
            self.generation = generation;
            
            // Evolve population
            self.evolve_generation();
            
            // Track statistics
            let best_fitness = self.get_best_individual().fitness;
            let best_container = self.get_best_individual().container_size;
            let diversity = self.calculate_diversity();
            self.best_fitness_history.push(best_fitness);
            self.diversity_history.push(diversity);
            
            // Print progress
            if generation % 50 == 0 || generation < 10 {
                println!("Gen {}: Best fitness: {:.2}, Container: {:.4}, Diversity: {:.3}", 
                         generation, best_fitness, best_container, diversity);
            }
            
            // Early termination if no improvement
            if generation > 100 && 
               self.best_fitness_history.len() > 50 &&
               self.best_fitness_history.iter().rev().take(50).all(|&f| (f - best_fitness).abs() < 1.0) {
                println!("Converged at generation {}", generation);
                break;
            }
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
            .map(|i| theoretical_min * (1.2 + (i as f64 / self.population_size as f64) * 0.5))
            .collect();
        
        // Generate individuals in parallel
        let individuals: Vec<Individual> = container_sizes
            .par_iter()
            .enumerate()
            .filter_map(|(i, &container_size)| {
                self.generate_random_individual(container_size, i % 4) // Different strategies
            })
            .collect();
        
        if individuals.len() < self.population_size / 2 {
            return Err("Failed to generate sufficient initial population".into());
        }
        
        self.population = individuals;
        self.population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        
        Ok(())
    }
    
    fn generate_random_individual(&self, container_size: f64, strategy: usize) -> Option<Individual> {
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
                            if rng.gen_bool(0.5) { 0.0 } else { container_size - self.square_size }
                        } else {
                            rng.gen::<f64>() * (container_size - self.square_size)
                        };
                        let y = if rng.gen_bool(edge_bias) {
                            if rng.gen_bool(0.5) { 0.0 } else { container_size - self.square_size }
                        } else {
                            rng.gen::<f64>() * (container_size - self.square_size)
                        };
                        (x, y, 0.0)
                    }
                    2 => {
                        // Grid-like placement with noise
                        let grid_size = (self.num_squares as f64).sqrt().ceil() as usize;
                        let cell_size = container_size / grid_size as f64;
                        let grid_x = squares.len() % grid_size;
                        let grid_y = squares.len() / grid_size;
                        
                        let noise = cell_size * 0.3;
                        let x = (grid_x as f64 * cell_size + rng.gen_range(-noise..noise))
                            .max(0.0).min(container_size - self.square_size);
                        let y = (grid_y as f64 * cell_size + rng.gen_range(-noise..noise))
                            .max(0.0).min(container_size - self.square_size);
                        (x, y, 0.0)
                    }
                    _ => {
                        // Corner-first placement
                        let corners = [
                            (0.0, 0.0),
                            (container_size - self.square_size, 0.0),
                            (0.0, container_size - self.square_size),
                            (container_size - self.square_size, container_size - self.square_size),
                        ];
                        
                        if squares.len() < 4 {
                            let (x, y) = corners[squares.len()];
                            (x, y, 0.0)
                        } else {
                            let x = rng.gen::<f64>() * (container_size - self.square_size);
                            let y = rng.gen::<f64>() * (container_size - self.square_size);
                            (x, y, 0.0)
                        }
                    }
                };
                
                let square = Square::new(x, y, self.square_size, angle);
                
                if square.is_inside_container(container_size) && 
                   !squares.iter().any(|s| square.overlaps_with(s)) {
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
        self.population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
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
                self.mutate(&mut offspring);
            }
            
            offspring.generation = self.generation;
            offspring.repair(self.square_size);
            
            new_population.push(offspring);
        }
        
        self.population = new_population;
        
        // Adaptive mutation rate
        let diversity = self.calculate_diversity();
        if diversity < 0.1 {
            self.mutation_rate = (self.mutation_rate * 1.1).min(0.3);
        } else if diversity > 0.5 {
            self.mutation_rate = (self.mutation_rate * 0.9).max(0.05);
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
                while attempts < 20 && offspring_squares.iter().any(|s| new_square.overlaps_with(s)) {
                    new_square.x += rng.gen_range(-0.5..0.5);
                    new_square.y += rng.gen_range(-0.5..0.5);
                    new_square.x = new_square.x.max(0.0).min(parent1.container_size - self.square_size);
                    new_square.y = new_square.y.max(0.0).min(parent1.container_size - self.square_size);
                    attempts += 1;
                }
                
                if !offspring_squares.iter().any(|s| new_square.overlaps_with(s)) {
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
                
                if square.is_inside_container(parent1.container_size) &&
                   !offspring_squares.iter().any(|s| square.overlaps_with(s)) {
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
    
    fn mutate(&self, individual: &mut Individual) {
        let mut rng = thread_rng();
        
        match rng.gen_range(0..4) {
            0 => {
                // Position mutation
                if !individual.squares.is_empty() {
                    let idx = rng.gen_range(0..individual.squares.len());
                    let square = &mut individual.squares[idx];
                    
                    let max_delta = individual.container_size * 0.1;
                    let dx = rng.gen_range(-max_delta..max_delta);
                    let dy = rng.gen_range(-max_delta..max_delta);
                    
                    square.x = (square.x + dx).max(0.0).min(individual.container_size - self.square_size);
                    square.y = (square.y + dy).max(0.0).min(individual.container_size - self.square_size);
                }
            }
            1 => {
                // Angle mutation
                if self.allow_rotation && !individual.squares.is_empty() {
                    let idx = rng.gen_range(0..individual.squares.len());
                    let square = &mut individual.squares[idx];
                    
                    let angle_delta = rng.gen_range(-PI/8.0..PI/8.0);
                    square.angle = (square.angle + angle_delta) % (PI / 2.0);
                }
            }
            2 => {
                // Swap mutation
                if individual.squares.len() >= 2 {
                    let idx1 = rng.gen_range(0..individual.squares.len());
                    let idx2 = rng.gen_range(0..individual.squares.len());
                    
                    if idx1 != idx2 {
                        let pos1 = (individual.squares[idx1].x, individual.squares[idx1].y);
                        let pos2 = (individual.squares[idx2].x, individual.squares[idx2].y);
                        
                        individual.squares[idx1].x = pos2.0;
                        individual.squares[idx1].y = pos2.1;
                        individual.squares[idx2].x = pos1.0;
                        individual.squares[idx2].y = pos1.1;
                    }
                }
            }
            _ => {
                // Container size mutation
                let delta = rng.gen_range(-0.1..0.1);
                individual.container_size = (individual.container_size * (1.0 + delta))
                    .max(self.square_size * (self.num_squares as f64).sqrt());
            }
        }
        
        // Recalculate fitness
        individual.fitness = Individual::calculate_fitness(&individual.squares, individual.container_size);
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
        self.population.iter().min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap()
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
        println!("Valid solutions in population: {}/{}", valid_count, self.population.len());
    }
}