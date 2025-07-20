use crate::animation::*;
use crate::geometry::*;
use rand::prelude::*;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct IterationRecord {
    pub iteration: usize,
    pub container_size: f64,
    pub best_size: f64,
    pub temperature: f64,
    pub energy: f64,
    pub improvement: bool,
}

pub struct SquarePackingSolver {
    num_squares: usize,
    square_size: f64,
    allow_rotation: bool,
    best_solution: Option<Vec<Square>>,
    best_container_size: f64,
    iteration_history: Vec<IterationRecord>,
    rng: ThreadRng,
    animation_recorder: Option<AnimationRecorder>,
}

impl SquarePackingSolver {
    pub fn new(num_squares: usize, square_size: f64, allow_rotation: bool) -> Self {
        Self {
            num_squares,
            square_size,
            allow_rotation,
            best_solution: None,
            best_container_size: f64::INFINITY,
            iteration_history: Vec::new(),
            rng: thread_rng(),
            animation_recorder: None,
        }
    }

    pub fn with_animation_recording(mut self, frame_interval: usize) -> Self {
        self.animation_recorder = Some(AnimationRecorder::new(
            "Simulated Annealing".to_string(),
            self.num_squares,
            self.square_size,
            self.allow_rotation,
            frame_interval,
        ));
        self
    }

    pub fn solve(
        &mut self,
        max_iterations: usize,
    ) -> Result<(Vec<Square>, f64), Box<dyn std::error::Error>> {
        let theoretical_min = (self.num_squares as f64).sqrt() * self.square_size;
        let mut container_size = theoretical_min * 1.3;

        // Try to find an initial solution
        let mut current_solution = Vec::new();
        let mut attempts = 0;
        while current_solution.len() != self.num_squares && attempts < 20 {
            current_solution = self.generate_random_solution(container_size);
            if current_solution.len() != self.num_squares {
                container_size *= 1.1;
            }
            attempts += 1;
        }

        if current_solution.len() != self.num_squares {
            return Err("Failed to find initial solution".into());
        }

        self.best_solution = Some(current_solution.clone());
        self.best_container_size = container_size;

        // Record initial frame
        let initial_energy = self.calculate_energy(&current_solution, container_size);
        if let Some(ref mut recorder) = self.animation_recorder {
            let state = AlgorithmState::SimulatedAnnealing {
                temperature: 1.0,
                energy: initial_energy,
                acceptance_rate: 0.0,
            };
            recorder.record_frame(0, &current_solution, container_size, state, false, Some("Initial solution".to_string()));
        }

        // Simulated annealing
        let mut temperature = 1.0;
        let cooling_rate = 0.9995;
        let min_temperature = 1e-8;
        let mut acceptance_count = 0;
        let mut total_attempts = 0;

        for iteration in 0..max_iterations {
            // Occasionally try to reduce container size
            if iteration % 500 == 0 && iteration > 1000 {
                let test_size = container_size * 0.995;
                if let Some(test_solution) = self.try_fit_in_container(&current_solution, test_size)
                {
                    container_size = test_size;
                    current_solution = test_solution;

                    if container_size < self.best_container_size {
                        self.best_container_size = container_size;
                        self.best_solution = Some(current_solution.clone());
                    }
                }
            }

            // Generate neighbor solution
            let new_solution = self.generate_neighbor(&current_solution, container_size);

            // Calculate energies (lower is better)
            let current_energy = self.calculate_energy(&current_solution, container_size);
            let new_energy = self.calculate_energy(&new_solution, container_size);

            let delta_energy = new_energy - current_energy;
            let accept = delta_energy < 0.0
                || (temperature > min_temperature
                    && self.rng.gen::<f64>() < (-delta_energy / temperature).exp());

            let mut improvement = false;
            if accept {
                current_solution = new_solution;
                acceptance_count += 1;

                // Check if this is a new best
                if container_size < self.best_container_size {
                    self.best_container_size = container_size;
                    self.best_solution = Some(current_solution.clone());
                    improvement = true;
                }
            }
            total_attempts += 1;

            // Record iteration
            if iteration % 100 == 0 {
                self.iteration_history.push(IterationRecord {
                    iteration,
                    container_size,
                    best_size: self.best_container_size,
                    temperature,
                    energy: current_energy,
                    improvement,
                });
            }

            // Record animation frame
            if let Some(ref mut recorder) = self.animation_recorder {
                let acceptance_rate = if total_attempts > 0 {
                    acceptance_count as f64 / total_attempts as f64
                } else {
                    0.0
                };

                let state = AlgorithmState::SimulatedAnnealing {
                    temperature,
                    energy: current_energy,
                    acceptance_rate,
                };

                if recorder.should_record_frame(iteration) {
                    recorder.record_frame(
                        iteration,
                        &current_solution,
                        container_size,
                        state.clone(),
                        improvement,
                        None,
                    );
                }

                // Record significant events
                if improvement {
                    recorder.record_significant_event(
                        iteration,
                        &current_solution,
                        container_size,
                        state,
                        &format!("New best solution: {:.6}", self.best_container_size),
                    );
                }
            }

            // Cool down
            temperature *= cooling_rate;
            temperature = temperature.max(min_temperature);

            // Early termination if we haven't improved in a while
            if iteration > 5000
                && self.iteration_history.len() > 10
                && self
                    .iteration_history
                    .iter()
                    .rev()
                    .take(10)
                    .all(|r| !r.improvement)
            {
                break;
            }
        }

        // Finalize animation recording
        if let Some(ref mut recorder) = self.animation_recorder {
            recorder.finalize();
        }

        match &self.best_solution {
            Some(solution) => Ok((solution.clone(), self.best_container_size)),
            None => Err("No solution found".into()),
        }
    }

    fn generate_random_solution(&mut self, container_size: f64) -> Vec<Square> {
        let mut squares = Vec::new();
        let max_attempts_per_square = 2000;

        for _ in 0..self.num_squares {
            let mut placed = false;

            for _ in 0..max_attempts_per_square {
                let x = self.rng.gen::<f64>() * (container_size - self.square_size);
                let y = self.rng.gen::<f64>() * (container_size - self.square_size);
                let angle = if self.allow_rotation {
                    self.rng.gen::<f64>() * PI / 2.0
                } else {
                    0.0
                };

                let square = Square::new(x, y, self.square_size, angle);

                if self.is_valid_placement(&square, &squares, container_size) {
                    squares.push(square);
                    placed = true;
                    break;
                }
            }

            if !placed {
                break; // Failed to place this square
            }
        }

        squares
    }

    fn generate_neighbor(&mut self, solution: &[Square], container_size: f64) -> Vec<Square> {
        if solution.is_empty() {
            return Vec::new();
        }

        let mut new_solution = solution.to_vec();
        let mutation_types = if self.allow_rotation {
            vec!["position", "angle", "position_angle", "swap"]
        } else {
            vec!["position", "swap"]
        };

        let mutation_type = mutation_types.choose(&mut self.rng).unwrap();

        match *mutation_type {
            "position" => self.mutate_position(&mut new_solution, container_size),
            "angle" => self.mutate_angle(&mut new_solution, container_size),
            "position_angle" => {
                self.mutate_position(&mut new_solution, container_size);
                self.mutate_angle(&mut new_solution, container_size);
            }
            "swap" => self.mutate_swap(&mut new_solution, container_size),
            _ => {}
        }

        new_solution
    }

    fn mutate_position(&mut self, solution: &mut [Square], container_size: f64) {
        if solution.is_empty() {
            return;
        }

        let idx = self.rng.gen_range(0..solution.len());
        let old_x = solution[idx].x;
        let old_y = solution[idx].y;

        // Try small perturbation first
        let max_delta = (container_size * 0.1).min(1.0);
        let dx = self.rng.gen_range(-max_delta..max_delta);
        let dy = self.rng.gen_range(-max_delta..max_delta);

        solution[idx].x = (old_x + dx).max(0.0).min(container_size - self.square_size);
        solution[idx].y = (old_y + dy).max(0.0).min(container_size - self.square_size);

        // Check if the new position is valid
        let other_squares: Vec<Square> = solution
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, s)| *s)
            .collect();

        if !self.is_valid_placement(&solution[idx], &other_squares, container_size) {
            // Revert if invalid
            solution[idx].x = old_x;
            solution[idx].y = old_y;
        }
    }

    fn mutate_angle(&mut self, solution: &mut [Square], container_size: f64) {
        if solution.is_empty() || !self.allow_rotation {
            return;
        }

        let idx = self.rng.gen_range(0..solution.len());
        let old_angle = solution[idx].angle;

        let angle_delta = self.rng.gen_range(-PI / 8.0..PI / 8.0);
        solution[idx].angle = (old_angle + angle_delta) % (PI / 2.0);

        let other_squares: Vec<Square> = solution
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, s)| *s)
            .collect();

        if !self.is_valid_placement(&solution[idx], &other_squares, container_size) {
            solution[idx].angle = old_angle;
        }
    }

    fn mutate_swap(&mut self, solution: &mut [Square], _container_size: f64) {
        if solution.len() < 2 {
            return;
        }

        let idx1 = self.rng.gen_range(0..solution.len());
        let idx2 = self.rng.gen_range(0..solution.len());

        if idx1 != idx2 {
            let pos1 = (solution[idx1].x, solution[idx1].y);
            let pos2 = (solution[idx2].x, solution[idx2].y);

            solution[idx1].x = pos2.0;
            solution[idx1].y = pos2.1;
            solution[idx2].x = pos1.0;
            solution[idx2].y = pos1.1;
        }
    }

    fn is_valid_placement(
        &self,
        square: &Square,
        existing_squares: &[Square],
        container_size: f64,
    ) -> bool {
        square.is_inside_container(container_size)
            && !existing_squares.iter().any(|s| square.overlaps_with(s))
    }

    fn calculate_energy(&self, solution: &[Square], container_size: f64) -> f64 {
        if solution.len() != self.num_squares {
            return f64::INFINITY;
        }

        let mut energy = 0.0;

        // Primary energy: container size (want to minimize)
        energy += container_size * 1000.0;

        // Penalty for overlaps (should be zero in valid solutions)
        for (i, sq1) in solution.iter().enumerate() {
            for sq2 in solution.iter().skip(i + 1) {
                if sq1.overlaps_with(sq2) {
                    energy += 10000.0;
                }
            }

            // Penalty for being outside container
            if !sq1.is_inside_container(container_size) {
                energy += 10000.0;
            }
        }

        // Secondary energy: encourage squares to be closer to edges/corners (packing efficiency)
        for square in solution {
            let center = square.center();
            let dist_to_edge = (center.x.min(container_size - center.x))
                .min(center.y.min(container_size - center.y));
            energy -= dist_to_edge * 0.1; // Small reward for being close to edges
        }

        energy
    }

    fn try_fit_in_container(
        &mut self,
        solution: &[Square],
        new_container_size: f64,
    ) -> Option<Vec<Square>> {
        // Simple check: see if current solution fits in smaller container
        if solution
            .iter()
            .all(|s| s.is_inside_container(new_container_size))
        {
            return Some(solution.to_vec());
        }

        // Try to adjust positions to fit
        let mut adjusted = solution.to_vec();
        for square in &mut adjusted {
            if !square.is_inside_container(new_container_size) {
                // Try to move it inside
                let (min_pt, max_pt) = square.bounding_box();
                let width = max_pt.x - min_pt.x;
                let height = max_pt.y - min_pt.y;

                if width <= new_container_size && height <= new_container_size {
                    // Can fit, just need to reposition
                    square.x = square.x.max(0.0).min(new_container_size - width);
                    square.y = square.y.max(0.0).min(new_container_size - height);
                } else {
                    return None; // Can't fit even with repositioning
                }
            }
        }

        // Check for overlaps after adjustment
        for (i, sq1) in adjusted.iter().enumerate() {
            for sq2 in adjusted.iter().skip(i + 1) {
                if sq1.overlaps_with(sq2) {
                    return None; // Overlaps created by adjustment
                }
            }
        }

        Some(adjusted)
    }

    pub fn print_convergence_summary(&self) {
        if self.iteration_history.is_empty() {
            println!("No convergence data available");
            return;
        }

        println!("\nConvergence Summary:");
        println!(
            "{:<10} {:<12} {:<12} {:<12} {:<8}",
            "Iteration", "Current", "Best", "Temperature", "Improved"
        );
        println!("{}", "-".repeat(60));

        for record in self
            .iteration_history
            .iter()
            .step_by(self.iteration_history.len().max(10) / 10)
        {
            println!(
                "{:<10} {:<12.6} {:<12.6} {:<12.2e} {:<8}",
                record.iteration,
                record.container_size,
                record.best_size,
                record.temperature,
                if record.improvement { "✓" } else { "" }
            );
        }

        if let Some(last) = self.iteration_history.last() {
            println!(
                "\nFinal: Container size {:.6}, {} iterations",
                last.best_size, last.iteration
            );
        }
    }

    pub fn save_animation_data<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref recorder) = self.animation_recorder {
            recorder.save_to_file(path)?;
            println!("Animation data saved");
        } else {
            return Err("No animation data to save".into());
        }
        Ok(())
    }
}
