use crate::animation::*;
use crate::geometry::*;
use plotters::prelude::*;
use std::path::Path;

pub struct AnimationRenderer {
    width: u32,
    height: u32,
}

impl AnimationRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn render_frame_to_file<P: AsRef<Path>>(
        &self,
        frame: &AnimationFrame,
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(&output_path, (self.width, self.height)).into_drawing_area();
        root.fill(&WHITE)?;

        self.render_frame_content(&root, frame)?;
        root.present()?;
        Ok(())
    }

    pub fn render_interpolated_frame_to_file<P: AsRef<Path>>(
        &self,
        frame: &InterpolatedFrame,
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(&output_path, (self.width, self.height)).into_drawing_area();
        root.fill(&WHITE)?;

        self.render_interpolated_frame_content(&root, frame)?;
        root.present()?;
        Ok(())
    }

    fn render_frame_content(
        &self,
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        frame: &AnimationFrame,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let container_size = frame.container_size;

        // Calculate appropriate padding based on square size and container
        let square_size = if !frame.squares.is_empty() {
            frame.squares[0].size
        } else {
            1.0
        };
        let padding = (square_size * 0.5).max(container_size * 0.1);

        // Calculate coordinate system with proper scaling
        let chart_area = root.margin(20, 20, 20, 100);
        let mut chart = ChartBuilder::on(&chart_area)
            .caption(
                &format!(
                    "Iteration {}: Container Size {:.3}",
                    frame.iteration, container_size
                ),
                ("Arial", 24).into_font(),
            )
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                -padding..(container_size + padding),
                -padding..(container_size + padding),
            )?;

        chart.configure_mesh().draw()?;

        // Draw container boundary
        chart.draw_series(std::iter::once(Rectangle::new(
            [(0.0, 0.0), (container_size, container_size)],
            BLACK.stroke_width(2),
        )))?;

        // Draw squares
        for (i, square) in frame.squares.iter().enumerate() {
            let color = self.get_square_color(i, frame.squares.len());
            self.draw_square(&mut chart, square, color)?;
        }

        // Draw algorithm info
        self.draw_algorithm_info(&chart_area, &frame.algorithm_state, frame.improvement)?;

        Ok(())
    }

    fn render_interpolated_frame_content(
        &self,
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        frame: &InterpolatedFrame,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let container_size = frame.container_size;

        // Calculate appropriate padding based on square size and container
        let square_size = if !frame.squares.is_empty() {
            frame.squares[0].size
        } else {
            1.0
        };
        let padding = (square_size * 0.5).max(container_size * 0.1);

        // Calculate coordinate system with proper scaling
        let chart_area = root.margin(20, 20, 20, 100);
        let mut chart = ChartBuilder::on(&chart_area)
            .caption(
                &format!(
                    "Time {:.3}s: Container Size {:.3}",
                    frame.timestamp, container_size
                ),
                ("Arial", 24).into_font(),
            )
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                -padding..(container_size + padding),
                -padding..(container_size + padding),
            )?;

        chart.configure_mesh().draw()?;

        // Draw container boundary
        chart.draw_series(std::iter::once(Rectangle::new(
            [(0.0, 0.0), (container_size, container_size)],
            BLACK.stroke_width(2),
        )))?;

        // Draw squares
        for (i, square) in frame.squares.iter().enumerate() {
            let color = self.get_square_color(i, frame.squares.len());
            self.draw_square(&mut chart, square, color)?;
        }

        // Draw algorithm info
        self.draw_algorithm_info(&chart_area, &frame.algorithm_state, false)?;

        Ok(())
    }

    fn draw_square<DB: DrawingBackend>(
        &self,
        chart: &mut ChartContext<
            DB,
            Cartesian2d<
                plotters::coord::types::RangedCoordf64,
                plotters::coord::types::RangedCoordf64,
            >,
        >,
        square: &Square,
        color: RGBColor,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        <DB as DrawingBackend>::ErrorType: 'static,
    {
        let corners = square.corners();

        // Draw filled square
        let polygon_points: Vec<(f64, f64)> = corners.iter().map(|p| (p.x, p.y)).collect();
        chart.draw_series(std::iter::once(Polygon::new(
            polygon_points.clone(),
            color.mix(0.3).filled(),
        )))?;

        // Draw square outline
        chart.draw_series(std::iter::once(Polygon::new(
            polygon_points,
            color.stroke_width(2),
        )))?;

        // Draw center point
        let center = square.center();
        chart.draw_series(std::iter::once(Circle::new(
            (center.x, center.y),
            3,
            color.filled(),
        )))?;

        Ok(())
    }

    fn draw_algorithm_info(
        &self,
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        state: &AlgorithmState,
        improvement: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let info_area = root
            .margin(0, 0, 0, 0)
            .margin(20, 20, self.height as i32 - 80, 20);

        match state {
            AlgorithmState::SimulatedAnnealing {
                temperature,
                energy,
                acceptance_rate,
            } => {
                let info_text = format!(
                    "Temperature: {:.6}\nEnergy: {:.1}\nAcceptance Rate: {:.1}%{}",
                    temperature,
                    energy,
                    acceptance_rate * 100.0,
                    if improvement {
                        "\n🚀 IMPROVEMENT!"
                    } else {
                        ""
                    }
                );

                info_area.draw_text(
                    &info_text,
                    &TextStyle::from(("Arial", 14)).color(&BLACK),
                    (10, 10),
                )?;
            }
            AlgorithmState::GeneticAlgorithm {
                generation,
                best_fitness,
                diversity,
                population_size,
                mutation_rate,
            } => {
                let info_text = format!(
                    "Generation: {}\nBest Fitness: {:.6}\nDiversity: {:.3}\nPopulation: {}\nMutation Rate: {:.3}",
                    generation, best_fitness, diversity, population_size, mutation_rate
                );

                info_area.draw_text(
                    &info_text,
                    &TextStyle::from(("Arial", 14)).color(&BLACK),
                    (10, 10),
                )?;
            }
        }

        Ok(())
    }

    fn get_square_color(&self, index: usize, total: usize) -> RGBColor {
        let hue = (index as f64 / total as f64) * 360.0;
        self.hsv_to_rgb(hue, 0.7, 0.8)
    }

    fn hsv_to_rgb(&self, h: f64, s: f64, v: f64) -> RGBColor {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        RGBColor(
            ((r + m) * 255.0) as u8,
            ((g + m) * 255.0) as u8,
            ((b + m) * 255.0) as u8,
        )
    }
}

pub struct AnimationSequenceRenderer {
    renderer: AnimationRenderer,
    target_fps: f64,
    interpolate: bool,
}

impl AnimationSequenceRenderer {
    pub fn new(width: u32, height: u32, target_fps: f64, interpolate: bool) -> Self {
        Self {
            renderer: AnimationRenderer::new(width, height),
            target_fps,
            interpolate,
        }
    }

    pub fn render_animation_sequence<P: AsRef<Path>>(
        &self,
        animation_data: &AnimationData,
        output_dir: P,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        let mut frame_files = Vec::new();

        if self.interpolate {
            self.render_interpolated_sequence(animation_data, output_dir, &mut frame_files)?;
        } else {
            self.render_direct_sequence(animation_data, output_dir, &mut frame_files)?;
        }

        Ok(frame_files)
    }

    fn render_direct_sequence<P: AsRef<Path>>(
        &self,
        animation_data: &AnimationData,
        output_dir: P,
        frame_files: &mut Vec<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let output_dir = output_dir.as_ref();

        for (i, frame) in animation_data.frames.iter().enumerate() {
            let filename = format!("frame_{:06}.png", i);
            let filepath = output_dir.join(&filename);

            self.renderer.render_frame_to_file(frame, &filepath)?;
            frame_files.push(filename);

            if i % 10 == 0 {
                println!("Rendered frame {}/{}", i + 1, animation_data.frames.len());
            }
        }

        Ok(())
    }

    fn render_interpolated_sequence<P: AsRef<Path>>(
        &self,
        animation_data: &AnimationData,
        output_dir: P,
        frame_files: &mut Vec<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let output_dir = output_dir.as_ref();

        if animation_data.frames.is_empty() {
            return Ok(());
        }

        let total_duration = animation_data.metadata.total_duration;
        let frame_duration = 1.0 / self.target_fps;
        let total_output_frames = (total_duration / frame_duration).ceil() as usize;

        for i in 0..total_output_frames {
            let timestamp = i as f64 * frame_duration;

            if let Some(interpolated_frame) = animation_data.interpolate_frame(timestamp) {
                let filename = format!("frame_{:06}.png", i);
                let filepath = output_dir.join(&filename);

                self.renderer
                    .render_interpolated_frame_to_file(&interpolated_frame, &filepath)?;
                frame_files.push(filename);

                if i % 50 == 0 {
                    println!(
                        "Rendered interpolated frame {}/{}",
                        i + 1,
                        total_output_frames
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_animation_renderer_creation() {
        let renderer = AnimationRenderer::new(800, 600);
        assert_eq!(renderer.width, 800);
        assert_eq!(renderer.height, 600);
    }

    #[test]
    fn test_hsv_to_rgb_conversion() {
        let renderer = AnimationRenderer::new(100, 100);

        // Test red (0°)
        let red = renderer.hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(red.0, 255);
        assert_eq!(red.1, 0);
        assert_eq!(red.2, 0);

        // Test green (120°)
        let green = renderer.hsv_to_rgb(120.0, 1.0, 1.0);
        assert_eq!(green.0, 0);
        assert_eq!(green.1, 255);
        assert_eq!(green.2, 0);
    }
}
