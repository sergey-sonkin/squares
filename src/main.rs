use std::time::Instant;
use clap::{Parser, Subcommand};

mod geometry;
mod solver;
mod genetic;
mod visualization;

use solver::*;
use genetic::*;
use visualization::*;

#[derive(Parser)]
#[command(name = "square-packing")]
#[command(about = "A high-performance square packing solver")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve square packing for N squares using simulated annealing
    Solve {
        /// Number of squares to pack
        #[arg(short, long)]
        num_squares: usize,
        
        /// Maximum iterations for optimization
        #[arg(short, long, default_value = "50000")]
        iterations: usize,
        
        /// Generate visualization
        #[arg(short, long)]
        visualize: bool,
        
        /// Allow rotation of squares
        #[arg(short, long)]
        rotation: bool,
    },
    
    /// Solve square packing using genetic algorithm
    Genetic {
        /// Number of squares to pack
        #[arg(short, long)]
        num_squares: usize,
        
        /// Maximum generations for evolution
        #[arg(short, long, default_value = "500")]
        generations: usize,
        
        /// Generate visualization
        #[arg(short, long)]
        visualize: bool,
        
        /// Allow rotation of squares
        #[arg(short, long)]
        rotation: bool,
    },
    
    /// Test against known optimal solutions
    Test {
        /// Maximum number of squares to test
        #[arg(short, long, default_value = "10")]
        max_squares: usize,
    },
    
    /// Benchmark the solver performance
    Benchmark {
        /// Number of squares for benchmark
        #[arg(short, long, default_value = "17")]
        num_squares: usize,
        
        /// Number of runs for averaging
        #[arg(short, long, default_value = "5")]
        runs: usize,
    },
    
    /// Compare genetic algorithm vs simulated annealing
    Compare {
        /// Number of squares to pack
        #[arg(short, long)]
        num_squares: usize,
        
        /// Number of runs for each algorithm
        #[arg(long, default_value = "3")]
        runs: usize,
        
        /// Allow rotation of squares
        #[arg(short, long)]
        rotation: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Solve { num_squares, iterations, visualize, rotation } => {
            println!("Solving {} squares packing problem...", num_squares);
            let start = Instant::now();
            
            let mut solver = SquarePackingSolver::new(num_squares, 1.0, rotation);
            let (solution, container_size) = solver.solve(iterations)?;
            
            let duration = start.elapsed();
            
            println!("Solution found in {:.2?}", duration);
            println!("Container size: {:.6}", container_size);
            println!("Efficiency: {:.2}%", (num_squares as f64 / container_size.powi(2)) * 100.0);
            
            if num_squares == 17 {
                println!("Bidwell's 1997 solution: 4.675");
                println!("Our ratio to Bidwell: {:.3}", container_size / 4.675);
            }
            
            if visualize {
                create_visualization(&solution, container_size, &format!("{}_squares_solution.png", num_squares))?;
                println!("Visualization saved to {}_squares_solution.png", num_squares);
            }
            
            // Show iteration history
            solver.print_convergence_summary();
        }
        
        Commands::Genetic { num_squares, generations, visualize, rotation } => {
            println!("Solving {} squares packing problem using genetic algorithm...", num_squares);
            let start = Instant::now();
            
            let mut genetic_solver = GeneticSolver::new(num_squares, 1.0, rotation);
            let (solution, container_size) = genetic_solver.solve(generations)?;
            
            let duration = start.elapsed();
            
            println!("Solution found in {:.2?}", duration);
            println!("Container size: {:.6}", container_size);
            println!("Efficiency: {:.2}%", (num_squares as f64 / container_size.powi(2)) * 100.0);
            
            if num_squares == 17 {
                println!("Bidwell's 1997 solution: 4.675");
                println!("Our ratio to Bidwell: {:.3}", container_size / 4.675);
            }
            
            if visualize {
                create_visualization(&solution, container_size, &format!("{}_squares_genetic.png", num_squares))?;
                println!("Visualization saved to {}_squares_genetic.png", num_squares);
            }
            
            // Show genetic algorithm statistics
            genetic_solver.print_statistics();
        }
        
        Commands::Test { max_squares } => {
            test_known_solutions(max_squares)?;
        }
        
        Commands::Benchmark { num_squares, runs } => {
            benchmark_solver(num_squares, runs)?;
        }
        
        Commands::Compare { num_squares, runs, rotation } => {
            compare_algorithms(num_squares, runs, rotation)?;
        }
    }
    
    Ok(())
}

fn test_known_solutions(max_squares: usize) -> Result<(), Box<dyn std::error::Error>> {
    let known_solutions = vec![
        (2, 2.0),
        (3, 2.0),
        (4, 2.0),
        (5, 2.707),
        (6, 3.0),
        (7, 3.0),
        (8, 3.0),
        (9, 3.0),
        (10, 3.162),
        (17, 4.675),
    ];
    
    println!("Testing against known optimal solutions:");
    println!("{:<4} {:<8} {:<8} {:<8} {:<8}", "N", "Known", "Found", "Ratio", "Time");
    println!("{}", "-".repeat(40));
    
    for (n, known) in known_solutions.iter().filter(|(n, _)| *n <= max_squares) {
        let start = Instant::now();
        let mut solver = SquarePackingSolver::new(*n, 1.0, true);
        
        match solver.solve(10000) {
            Ok((_, found_size)) => {
                let duration = start.elapsed();
                let ratio = found_size / known;
                println!("{:<4} {:<8.3} {:<8.3} {:<8.3} {:<8.0}ms", 
                        n, known, found_size, ratio, duration.as_millis());
            }
            Err(_) => {
                println!("{:<4} {:<8.3} FAILED", n, known);
            }
        }
    }
    
    Ok(())
}

fn benchmark_solver(num_squares: usize, runs: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking solver with {} squares over {} runs...", num_squares, runs);
    
    let mut times = Vec::new();
    let mut solutions = Vec::new();
    
    for run in 1..=runs {
        print!("Run {}/{}: ", run, runs);
        let start = Instant::now();
        
        let mut solver = SquarePackingSolver::new(num_squares, 1.0, true);
        match solver.solve(20000) {
            Ok((_, size)) => {
                let duration = start.elapsed();
                times.push(duration.as_millis());
                solutions.push(size);
                println!("Container size: {:.6}, Time: {}ms", size, duration.as_millis());
            }
            Err(e) => {
                println!("Failed: {}", e);
            }
        }
    }
    
    if !solutions.is_empty() {
        let avg_time = times.iter().sum::<u128>() as f64 / times.len() as f64;
        let best_solution = solutions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let avg_solution = solutions.iter().sum::<f64>() / solutions.len() as f64;
        
        println!("\nBenchmark Results:");
        println!("Average time: {:.1}ms", avg_time);
        println!("Best solution: {:.6}", best_solution);
        println!("Average solution: {:.6}", avg_solution);
        println!("Standard deviation: {:.6}", 
                 (solutions.iter().map(|x| (x - avg_solution).powi(2)).sum::<f64>() / solutions.len() as f64).sqrt());
    }
    
    Ok(())
}

fn compare_algorithms(num_squares: usize, runs: usize, rotation: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing GA vs SA for {} squares over {} runs each...\n", num_squares, runs);
    
    let mut sa_times = Vec::new();
    let mut sa_solutions = Vec::new();
    let mut ga_times = Vec::new();
    let mut ga_solutions = Vec::new();
    
    // Test Simulated Annealing
    println!("Testing Simulated Annealing:");
    for run in 1..=runs {
        print!("  Run {}/{}: ", run, runs);
        let start = Instant::now();
        
        let mut solver = SquarePackingSolver::new(num_squares, 1.0, rotation);
        match solver.solve(20000) {
            Ok((_, size)) => {
                let duration = start.elapsed();
                sa_times.push(duration.as_millis());
                sa_solutions.push(size);
                println!("Container: {:.6}, Time: {}ms", size, duration.as_millis());
            }
            Err(e) => {
                println!("Failed: {}", e);
            }
        }
    }
    
    // Test Genetic Algorithm
    println!("\nTesting Genetic Algorithm:");
    for run in 1..=runs {
        print!("  Run {}/{}: ", run, runs);
        let start = Instant::now();
        
        let mut genetic_solver = GeneticSolver::new(num_squares, 1.0, rotation);
        match genetic_solver.solve(300) {
            Ok((_, size)) => {
                let duration = start.elapsed();
                ga_times.push(duration.as_millis());
                ga_solutions.push(size);
                println!("Container: {:.6}, Time: {}ms", size, duration.as_millis());
            }
            Err(e) => {
                println!("Failed: {}", e);
            }
        }
    }
    
    // Compare results
    println!("\n{:<20} {:<15} {:<15} {:<15} {:<15}", "Algorithm", "Best", "Average", "Avg Time", "Improvement");
    println!("{}", "-".repeat(80));
    
    if !sa_solutions.is_empty() {
        let sa_best = sa_solutions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let sa_avg = sa_solutions.iter().sum::<f64>() / sa_solutions.len() as f64;
        let sa_avg_time = sa_times.iter().sum::<u128>() as f64 / sa_times.len() as f64;
        
        println!("{:<20} {:<15.6} {:<15.6} {:<15.0}ms {:<15}", 
                "Simulated Annealing", sa_best, sa_avg, sa_avg_time, "-");
    }
    
    if !ga_solutions.is_empty() {
        let ga_best = ga_solutions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ga_avg = ga_solutions.iter().sum::<f64>() / ga_solutions.len() as f64;
        let ga_avg_time = ga_times.iter().sum::<u128>() as f64 / ga_times.len() as f64;
        
        let improvement = if !sa_solutions.is_empty() {
            let sa_avg = sa_solutions.iter().sum::<f64>() / sa_solutions.len() as f64;
            format!("{:.1}%", ((sa_avg - ga_avg) / sa_avg) * 100.0)
        } else {
            "-".to_string()
        };
        
        println!("{:<20} {:<15.6} {:<15.6} {:<15.0}ms {:<15}", 
                "Genetic Algorithm", ga_best, ga_avg, ga_avg_time, improvement);
    }
    
    // Show comparison to known optimals
    if num_squares == 17 {
        println!("\nComparison to Bidwell's 1997 optimal (4.675):");
        if !sa_solutions.is_empty() {
            let sa_best = sa_solutions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            println!("  SA best ratio: {:.3}x", sa_best / 4.675);
        }
        if !ga_solutions.is_empty() {
            let ga_best = ga_solutions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            println!("  GA best ratio: {:.3}x", ga_best / 4.675);
        }
    }
    
    Ok(())
}
