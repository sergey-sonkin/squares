use crate::geometry::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationFrame {
    pub iteration: usize,
    pub timestamp: f64,
    pub squares: Vec<Square>,
    pub container_size: f64,
    pub algorithm_state: AlgorithmState,
    pub improvement: bool,
    pub significant_event: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmState {
    SimulatedAnnealing {
        temperature: f64,
        energy: f64,
        acceptance_rate: f64,
    },
    GeneticAlgorithm {
        generation: usize,
        best_fitness: f64,
        diversity: f64,
        population_size: usize,
        mutation_rate: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationData {
    pub algorithm_name: String,
    pub problem_size: usize,
    pub square_size: f64,
    pub allow_rotation: bool,
    pub frames: Vec<AnimationFrame>,
    pub metadata: AnimationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationMetadata {
    pub start_time: String,
    pub total_duration: f64,
    pub final_container_size: f64,
    pub total_iterations: usize,
    pub convergence_point: Option<usize>,
    pub parameters: serde_json::Value,
}

pub struct AnimationRecorder {
    pub data: AnimationData,
    pub recording: bool,
    pub frame_interval: usize,
    pub last_recorded_iteration: usize,
    start_time: std::time::Instant,
}

impl AnimationRecorder {
    pub fn new(
        algorithm_name: String,
        problem_size: usize,
        square_size: f64,
        allow_rotation: bool,
        frame_interval: usize,
    ) -> Self {
        let data = AnimationData {
            algorithm_name,
            problem_size,
            square_size,
            allow_rotation,
            frames: Vec::new(),
            metadata: AnimationMetadata {
                start_time: chrono::Utc::now().to_rfc3339(),
                total_duration: 0.0,
                final_container_size: 0.0,
                total_iterations: 0,
                convergence_point: None,
                parameters: serde_json::json!({}),
            },
        };

        Self {
            data,
            recording: true,
            frame_interval,
            last_recorded_iteration: 0,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn should_record_frame(&self, iteration: usize) -> bool {
        self.recording && (
            iteration == 0 || // Always record first frame
            iteration % self.frame_interval == 0 || // Regular intervals
            iteration - self.last_recorded_iteration >= self.frame_interval
        )
    }

    pub fn record_frame(
        &mut self,
        iteration: usize,
        squares: &[Square],
        container_size: f64,
        algorithm_state: AlgorithmState,
        improvement: bool,
        significant_event: Option<String>,
    ) {
        if !self.recording {
            return;
        }

        let timestamp = self.start_time.elapsed().as_secs_f64();

        let frame = AnimationFrame {
            iteration,
            timestamp,
            squares: squares.to_vec(),
            container_size,
            algorithm_state,
            improvement,
            significant_event,
        };

        self.data.frames.push(frame);
        self.last_recorded_iteration = iteration;

        // Update metadata
        self.data.metadata.total_duration = timestamp;
        self.data.metadata.final_container_size = container_size;
        self.data.metadata.total_iterations = iteration;
    }

    pub fn record_significant_event(
        &mut self,
        iteration: usize,
        squares: &[Square],
        container_size: f64,
        algorithm_state: AlgorithmState,
        event_description: &str,
    ) {
        // Always record significant events regardless of interval
        let timestamp = self.start_time.elapsed().as_secs_f64();

        let frame = AnimationFrame {
            iteration,
            timestamp,
            squares: squares.to_vec(),
            container_size,
            algorithm_state,
            improvement: true,
            significant_event: Some(event_description.to_string()),
        };

        self.data.frames.push(frame);
    }

    pub fn set_convergence_point(&mut self, iteration: usize) {
        self.data.metadata.convergence_point = Some(iteration);
    }

    pub fn set_parameters(&mut self, parameters: serde_json::Value) {
        self.data.metadata.parameters = parameters;
    }

    pub fn finalize(&mut self) {
        self.recording = false;
        self.data.metadata.total_duration = self.start_time.elapsed().as_secs_f64();
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.data)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<AnimationData, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let data: AnimationData = serde_json::from_str(&json)?;
        Ok(data)
    }

    pub fn get_frame_count(&self) -> usize {
        self.data.frames.len()
    }

    pub fn get_duration(&self) -> f64 {
        self.data.metadata.total_duration
    }

    pub fn get_fps(&self) -> f64 {
        if self.data.metadata.total_duration > 0.0 {
            self.data.frames.len() as f64 / self.data.metadata.total_duration
        } else {
            0.0
        }
    }
}

// Helper functions for animation analysis
impl AnimationData {
    pub fn get_improvement_frames(&self) -> Vec<&AnimationFrame> {
        self.frames.iter().filter(|f| f.improvement).collect()
    }

    pub fn get_significant_events(&self) -> Vec<&AnimationFrame> {
        self.frames.iter().filter(|f| f.significant_event.is_some()).collect()
    }

    pub fn get_container_size_progression(&self) -> Vec<(f64, f64)> {
        self.frames.iter()
            .map(|f| (f.timestamp, f.container_size))
            .collect()
    }

    pub fn get_frame_at_time(&self, timestamp: f64) -> Option<&AnimationFrame> {
        self.frames.iter()
            .min_by(|a, b| (a.timestamp - timestamp).abs().partial_cmp(&(b.timestamp - timestamp).abs()).unwrap())
    }

    pub fn interpolate_frame(&self, timestamp: f64) -> Option<InterpolatedFrame> {
        if self.frames.is_empty() {
            return None;
        }

        // Find surrounding frames
        let mut before = None;
        let mut after = None;

        for frame in &self.frames {
            if frame.timestamp <= timestamp {
                before = Some(frame);
            } else if after.is_none() {
                after = Some(frame);
                break;
            }
        }

        match (before, after) {
            (Some(before_frame), Some(after_frame)) => {
                // Interpolate between frames
                let duration = after_frame.timestamp - before_frame.timestamp;
                if duration == 0.0 {
                    return Some(InterpolatedFrame::from_frame(before_frame));
                }

                let progress = (timestamp - before_frame.timestamp) / duration;
                Some(InterpolatedFrame::interpolate(before_frame, after_frame, progress))
            }
            (Some(frame), None) => {
                // Only before frame exists (end of animation)
                Some(InterpolatedFrame::from_frame(frame))
            }
            (None, Some(frame)) => {
                // Only after frame exists (start of animation)
                Some(InterpolatedFrame::from_frame(frame))
            }
            (None, None) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InterpolatedFrame {
    pub timestamp: f64,
    pub squares: Vec<Square>,
    pub container_size: f64,
    pub algorithm_state: AlgorithmState,
}

impl InterpolatedFrame {
    pub fn from_frame(frame: &AnimationFrame) -> Self {
        Self {
            timestamp: frame.timestamp,
            squares: frame.squares.clone(),
            container_size: frame.container_size,
            algorithm_state: frame.algorithm_state.clone(),
        }
    }

    pub fn interpolate(before: &AnimationFrame, after: &AnimationFrame, progress: f64) -> Self {
        let progress = progress.clamp(0.0, 1.0);

        // Interpolate container size
        let container_size = before.container_size + (after.container_size - before.container_size) * progress;

        // Interpolate square positions (assuming same number of squares)
        let mut squares = Vec::new();
        let min_count = before.squares.len().min(after.squares.len());

        for i in 0..min_count {
            let before_sq = &before.squares[i];
            let after_sq = &after.squares[i];

            let x = before_sq.x + (after_sq.x - before_sq.x) * progress;
            let y = before_sq.y + (after_sq.y - before_sq.y) * progress;
            
            // Handle angle interpolation (shortest path)
            let mut angle_diff = after_sq.angle - before_sq.angle;
            if angle_diff > PI / 4.0 {
                angle_diff -= PI / 2.0;
            } else if angle_diff < -PI / 4.0 {
                angle_diff += PI / 2.0;
            }
            let angle = before_sq.angle + angle_diff * progress;

            squares.push(Square::new(x, y, before_sq.size, angle));
        }

        // For algorithm state, use the "after" state for simplicity
        // (Could be made more sophisticated)
        let algorithm_state = after.algorithm_state.clone();

        Self {
            timestamp: before.timestamp + (after.timestamp - before.timestamp) * progress,
            squares,
            container_size,
            algorithm_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_animation_recorder_creation() {
        let recorder = AnimationRecorder::new(
            "TestAlgorithm".to_string(),
            10,
            1.0,
            true,
            100,
        );

        assert_eq!(recorder.data.algorithm_name, "TestAlgorithm");
        assert_eq!(recorder.data.problem_size, 10);
        assert!(recorder.recording);
    }

    #[test]
    fn test_frame_recording() {
        let mut recorder = AnimationRecorder::new(
            "TestAlgorithm".to_string(),
            2,
            1.0,
            false,
            1,
        );

        let squares = vec![
            Square::new(0.0, 0.0, 1.0, 0.0),
            Square::new(1.5, 0.0, 1.0, 0.0),
        ];

        let state = AlgorithmState::SimulatedAnnealing {
            temperature: 1.0,
            energy: 100.0,
            acceptance_rate: 0.5,
        };

        recorder.record_frame(0, &squares, 3.0, state, false, None);
        assert_eq!(recorder.data.frames.len(), 1);
        assert_eq!(recorder.data.frames[0].squares.len(), 2);
    }

    #[test]
    fn test_frame_interpolation() {
        let frame1 = AnimationFrame {
            iteration: 0,
            timestamp: 0.0,
            squares: vec![Square::new(0.0, 0.0, 1.0, 0.0)],
            container_size: 3.0,
            algorithm_state: AlgorithmState::SimulatedAnnealing {
                temperature: 1.0,
                energy: 100.0,
                acceptance_rate: 0.5,
            },
            improvement: false,
            significant_event: None,
        };

        let frame2 = AnimationFrame {
            iteration: 100,
            timestamp: 1.0,
            squares: vec![Square::new(2.0, 1.0, 1.0, 0.0)],
            container_size: 2.5,
            algorithm_state: AlgorithmState::SimulatedAnnealing {
                temperature: 0.5,
                energy: 50.0,
                acceptance_rate: 0.3,
            },
            improvement: true,
            significant_event: None,
        };

        let interpolated = InterpolatedFrame::interpolate(&frame1, &frame2, 0.5);
        
        assert_eq!(interpolated.squares.len(), 1);
        assert!((interpolated.squares[0].x - 1.0).abs() < 1e-10);
        assert!((interpolated.squares[0].y - 0.5).abs() < 1e-10);
        assert!((interpolated.container_size - 2.75).abs() < 1e-10);
    }
}