use std::time::Instant;
use clap::{Parser, Subcommand};

mod geometry;
mod solver;
mod visualization;

use solver::*;
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
    /// Solve square packing for N squares
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
        
        Commands::Test { max_squares } => {
            test_known_solutions(max_squares)?;
        }
        
        Commands::Benchmark { num_squares, runs } => {
            benchmark_solver(num_squares, runs)?;
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
