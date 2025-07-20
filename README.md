# Square Packing Solver

A high-performance square packing optimization suite with advanced animation and visualization capabilities.

## Features

- **Dual Algorithm Support**: Simulated Annealing and Genetic Algorithm solvers
- **Animation System**: Record and render optimization process as smooth videos
- **Output Organization**: Structured folders for JSON, videos, images, and frames
- **Multiple Formats**: MP4, GIF, WEBM video generation with FFmpeg
- **SAT Collision Detection**: Precise overlap detection for rotated squares
- **Real-time Visualization**: Progress tracking and solution display
- **Comprehensive Benchmarking**: Performance comparison and testing tools

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

### Animation & Video Generation
```bash
# Record simulated annealing with animation
cargo run -- solve --num-squares 8 --rotation --record-animation sa_demo --frame-interval 50 --visualize

# Record genetic algorithm with animation  
cargo run -- genetic --num-squares 8 --generations 200 --rotation --record-animation ga_demo --frame-interval 10 --visualize

# Generate MP4 video from animation data
cargo run -- animate -i ga_demo -o genetic_demo.mp4 --fps 24 --interpolate

# Record AND generate video
cargo run -- genetic --num-squares 8 --generations 200 --rotation --record-animation ga_demo --frame-interval 10 --generate-video genetic_demo.mp4 --video-fps 24 --video-interpolate --visualize

# Export frames for custom editing
cargo run -- render-animation -i sa_demo -o sa_frames --fps 30 --interpolate
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

| Algorithm | 17 Squares | Time | Animation Support |
|-----------|------------|------|-------------------|
| Simulated Annealing | 1.35x optimal | ~2.5s | ✅ Full recording |
| **Genetic Algorithm** | **1.06x optimal** | ~15s | ✅ Full recording |

- Bidwell's 1997 optimal: 4.675
- Genetic algorithm finds solutions only 6% above optimal
- Supports square rotation for better packing efficiency
- Complete animation pipeline: optimization → JSON → MP4/GIF/WEBM

## Output Structure

```
outputs/
├── json/           # Animation data files
├── mp4/            # Video animations  
├── gif/            # GIF animations
├── webm/           # WebM videos
├── frames/         # Frame sequences
└── images/         # Static visualizations
```

## Animation Features

- **Smooth Interpolation**: 60+ FPS frame interpolation for fluid motion
- **Algorithm Visualization**: Real-time display of optimization metrics
- **Multiple Formats**: MP4, GIF, WebM with configurable quality
- **Organized Output**: Automatic file organization by type
- **FFmpeg Integration**: Professional video encoding with compression
- **Educational Value**: Clear demonstration of algorithm differences
