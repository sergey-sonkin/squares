use clap::{Parser, Subcommand};
use std::time::Instant;

mod animation;
mod animation_renderer;
mod genetic;
mod geometry;
mod output_manager;
mod solver;
mod video_generator;
mod visualization;

use animation::*;
use animation_renderer::*;
use genetic::*;
use output_manager::*;
use solver::*;
use video_generator::*;
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

        /// Record animation data to file
        #[arg(long)]
        record_animation: Option<String>,

        /// Animation frame recording interval
        #[arg(long, default_value = "100")]
        frame_interval: usize,
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

        /// Record animation data to file
        #[arg(long)]
        record_animation: Option<String>,

        /// Animation frame recording interval
        #[arg(long, default_value = "10")]
        frame_interval: usize,
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

    /// Render animation from recorded data
    RenderAnimation {
        /// Path to animation data file
        #[arg(short, long)]
        input: String,

        /// Output directory for frames
        #[arg(short, long, default_value = "animation_frames")]
        output_dir: String,

        /// Target frames per second
        #[arg(long, default_value = "30")]
        fps: f64,

        /// Frame width in pixels
        #[arg(long, default_value = "800")]
        width: u32,

        /// Frame height in pixels
        #[arg(long, default_value = "600")]
        height: u32,

        /// Enable interpolation between frames
        #[arg(long)]
        interpolate: bool,
    },

    /// Generate video animation from recorded data
    Animate {
        /// Path to animation data file
        #[arg(short, long)]
        input: String,

        /// Output video file path
        #[arg(short, long)]
        output: String,

        /// Target frames per second
        #[arg(long, default_value = "30")]
        fps: f64,

        /// Video width in pixels
        #[arg(long, default_value = "800")]
        width: u32,

        /// Video height in pixels
        #[arg(long, default_value = "600")]
        height: u32,

        /// Enable interpolation between frames
        #[arg(long)]
        interpolate: bool,

        /// Video quality (low, medium, high)
        #[arg(long, default_value = "medium")]
        quality: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize output manager and create directories
    let output_manager = OutputManager::new();
    output_manager.create_directories()?;

    match cli.command {
        Commands::Solve {
            num_squares,
            iterations,
            visualize,
            rotation,
            record_animation,
            frame_interval,
        } => {
            println!("Solving {} squares packing problem...", num_squares);
            let start = Instant::now();

            let mut solver = SquarePackingSolver::new(num_squares, 1.0, rotation);
            
            // Enable animation recording if requested
            if record_animation.is_some() {
                solver = solver.with_animation_recording(frame_interval);
            }
            
            let (solution, container_size) = solver.solve(iterations)?;

            let duration = start.elapsed();

            println!("Solution found in {:.2?}", duration);
            println!("Container size: {:.6}", container_size);
            println!(
                "Efficiency: {:.2}%",
                (num_squares as f64 / container_size.powi(2)) * 100.0
            );

            if num_squares == 17 {
                println!("Bidwell's 1997 solution: 4.675");
                println!("Our ratio to Bidwell: {:.3}", container_size / 4.675);
            }

            if visualize {
                let vis_filename = format!("{}_squares_solution.png", num_squares);
                let vis_path = output_manager.image_path(&vis_filename);
                create_visualization(&solution, container_size, vis_path.to_str().unwrap())?;
                println!("Visualization saved to: {}", vis_path.display());
            }

            // Save animation data if requested
            if let Some(animation_file) = record_animation {
                let output_path = output_manager.animation_json_path(&animation_file);
                solver.save_animation_data(&output_path)?;
                println!("Animation data saved to: {}", output_path.display());
            }

            // Show iteration history
            solver.print_convergence_summary();
        }

        Commands::Genetic {
            num_squares,
            generations,
            visualize,
            rotation,
            record_animation,
            frame_interval,
        } => {
            println!(
                "Solving {} squares packing problem using genetic algorithm...",
                num_squares
            );
            let start = Instant::now();

            let mut genetic_solver = GeneticSolver::new(num_squares, 1.0, rotation);
            
            // Enable animation recording if requested
            if record_animation.is_some() {
                genetic_solver = genetic_solver.with_animation_recording(frame_interval);
            }
            
            let (solution, container_size) = genetic_solver.solve(generations)?;

            let duration = start.elapsed();

            println!("Solution found in {:.2?}", duration);
            println!("Container size: {:.6}", container_size);
            println!(
                "Efficiency: {:.2}%",
                (num_squares as f64 / container_size.powi(2)) * 100.0
            );

            if num_squares == 17 {
                println!("Bidwell's 1997 solution: 4.675");
                println!("Our ratio to Bidwell: {:.3}", container_size / 4.675);
            }

            if visualize {
                let vis_filename = format!("{}_squares_genetic.png", num_squares);
                let vis_path = output_manager.image_path(&vis_filename);
                create_visualization(&solution, container_size, vis_path.to_str().unwrap())?;
                println!("Visualization saved to: {}", vis_path.display());
            }

            // Save animation data if requested
            if let Some(animation_file) = record_animation {
                let output_path = output_manager.animation_json_path(&animation_file);
                genetic_solver.save_animation_data(&output_path)?;
                println!("Animation data saved to: {}", output_path.display());
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

        Commands::Compare {
            num_squares,
            runs,
            rotation,
        } => {
            compare_algorithms(num_squares, runs, rotation)?;
        }

        Commands::RenderAnimation {
            input,
            output_dir,
            fps,
            width,
            height,
            interpolate,
        } => {
            render_animation_command(input, output_dir, fps, width, height, interpolate)?;
        }

        Commands::Animate {
            input,
            output,
            fps,
            width,
            height,
            interpolate,
            quality,
        } => {
            animate_command(input, output, fps, width, height, interpolate, quality)?;
        }
    }

    Ok(())
}

fn render_animation_command(
    input_path: String,
    output_dir: String,
    fps: f64,
    width: u32,
    height: u32,
    interpolate: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_manager = OutputManager::new();
    
    // Handle input path - check if it's just a filename or full path
    let full_input_path = if input_path.contains('/') || input_path.contains('\\') {
        input_path.clone()
    } else {
        output_manager.animation_json_path(&input_path).to_string_lossy().to_string()
    };
    
    println!("Loading animation data from {}...", full_input_path);
    let animation_data = AnimationRecorder::load_from_file(&full_input_path)?;
    
    println!("Animation info:");
    println!("  Algorithm: {}", animation_data.algorithm_name);
    println!("  Problem size: {} squares", animation_data.problem_size);
    println!("  Total frames: {}", animation_data.frames.len());
    println!("  Duration: {:.2}s", animation_data.metadata.total_duration);
    
    let renderer = AnimationSequenceRenderer::new(width, height, fps, interpolate);
    
    // Use output manager for frames directory
    let frames_path = output_manager.frames_dir_path(&output_dir);
    println!("Rendering animation frames to {}...", frames_path.display());
    let frame_files = renderer.render_animation_sequence(&animation_data, &frames_path)?;
    
    println!("Successfully rendered {} frames", frame_files.len());
    println!("Output directory: {}", frames_path.display());
    
    if interpolate {
        println!("Interpolated to {:.1} FPS", fps);
        println!("Total output frames: {}", frame_files.len());
    }
    
    println!("\nTo create a video, you can use the animate command:");
    println!("  cargo run -- animate -i {} -o video_name.mp4", input_path);
    
    Ok(())
}

fn animate_command(
    input_path: String,
    output_path: String,
    fps: f64,
    width: u32,
    height: u32,
    interpolate: bool,
    quality: String,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check if FFmpeg is available
    if !detect_ffmpeg() {
        return Err("FFmpeg not found. Please install FFmpeg to generate videos.".into());
    }

    let output_manager = OutputManager::new();
    
    // Handle input path - check if it's just a filename or full path
    let full_input_path = if input_path.contains('/') || input_path.contains('\\') {
        input_path.clone()
    } else {
        output_manager.animation_json_path(&input_path).to_string_lossy().to_string()
    };

    println!("Loading animation data from {}...", full_input_path);
    let animation_data = AnimationRecorder::load_from_file(&full_input_path)?;
    
    println!("Animation info:");
    println!("  Algorithm: {}", animation_data.algorithm_name);
    println!("  Problem size: {} squares", animation_data.problem_size);
    println!("  Total frames: {}", animation_data.frames.len());
    println!("  Duration: {:.2}s", animation_data.metadata.total_duration);

    // Use output manager for organized output
    let final_output_path = output_manager.get_output_path(&output_path);
    
    // Determine video format from output extension
    let extension = final_output_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("mp4");
    
    let format = VideoFormat::from_extension(extension)
        .unwrap_or(VideoFormat::Mp4);

    // Parse quality setting
    let video_quality = match quality.to_lowercase().as_str() {
        "low" => VideoQuality::Low,
        "medium" => VideoQuality::Medium,
        "high" => VideoQuality::High,
        _ => {
            println!("Warning: Unknown quality '{}', using medium", quality);
            VideoQuality::Medium
        }
    };

    // Create video generator config
    let config = VideoGeneratorConfig {
        width,
        height,
        fps,
        format,
        interpolate,
        quality: video_quality,
    };

    // Estimate output size
    let generator = VideoGenerator::new(config);
    let size_estimate = generator.estimate_output_size(&animation_data)?;
    println!("{}", size_estimate);

    // Generate video
    println!("Generating {} video...", extension.to_uppercase());
    generator.generate_video(&animation_data, &final_output_path)?;
    
    println!("Video generation complete!");
    println!("Output file: {}", final_output_path.display());
    
    // Display file size
    if let Ok(metadata) = std::fs::metadata(&final_output_path) {
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("File size: {:.2} MB", size_mb);
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
    println!(
        "{:<4} {:<8} {:<8} {:<8} {:<8}",
        "N", "Known", "Found", "Ratio", "Time"
    );
    println!("{}", "-".repeat(40));

    for (n, known) in known_solutions.iter().filter(|(n, _)| *n <= max_squares) {
        let start = Instant::now();
        let mut solver = SquarePackingSolver::new(*n, 1.0, true);

        match solver.solve(10000) {
            Ok((_, found_size)) => {
                let duration = start.elapsed();
                let ratio = found_size / known;
                println!(
                    "{:<4} {:<8.3} {:<8.3} {:<8.3} {:<8.0}ms",
                    n,
                    known,
                    found_size,
                    ratio,
                    duration.as_millis()
                );
            }
            Err(_) => {
                println!("{:<4} {:<8.3} FAILED", n, known);
            }
        }
    }

    Ok(())
}

fn benchmark_solver(num_squares: usize, runs: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Benchmarking solver with {} squares over {} runs...",
        num_squares, runs
    );

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
                println!(
                    "Container size: {:.6}, Time: {}ms",
                    size,
                    duration.as_millis()
                );
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
        println!(
            "Standard deviation: {:.6}",
            (solutions
                .iter()
                .map(|x| (x - avg_solution).powi(2))
                .sum::<f64>()
                / solutions.len() as f64)
                .sqrt()
        );
    }

    Ok(())
}

fn compare_algorithms(
    num_squares: usize,
    runs: usize,
    rotation: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Comparing GA vs SA for {} squares over {} runs each...\n",
        num_squares, runs
    );

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
    println!(
        "\n{:<20} {:<15} {:<15} {:<15} {:<15}",
        "Algorithm", "Best", "Average", "Avg Time", "Improvement"
    );
    println!("{}", "-".repeat(80));

    if !sa_solutions.is_empty() {
        let sa_best = sa_solutions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let sa_avg = sa_solutions.iter().sum::<f64>() / sa_solutions.len() as f64;
        let sa_avg_time = sa_times.iter().sum::<u128>() as f64 / sa_times.len() as f64;

        println!(
            "{:<20} {:<15.6} {:<15.6} {:<15.0}ms {:<15}",
            "Simulated Annealing", sa_best, sa_avg, sa_avg_time, "-"
        );
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

        println!(
            "{:<20} {:<15.6} {:<15.6} {:<15.0}ms {:<15}",
            "Genetic Algorithm", ga_best, ga_avg, ga_avg_time, improvement
        );
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
