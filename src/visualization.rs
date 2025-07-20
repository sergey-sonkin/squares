use crate::geometry::*;
use plotters::prelude::*;
use std::error::Error;

pub fn create_visualization(
    squares: &[Square],
    container_size: f64,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            &format!(
                "Square Packing: {} squares in {:.4} container",
                squares.len(),
                container_size
            ),
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..container_size, 0.0..container_size)?;

    chart.configure_mesh().draw()?;

    // Draw container boundary
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (container_size, container_size)],
        BLACK.stroke_width(3).filled(),
    )))?;

    // Color palette for squares
    let colors = [
        &RED,
        &BLUE,
        &GREEN,
        &MAGENTA,
        &CYAN,
        &YELLOW,
        &(RGBColor(255, 165, 0)),   // Orange
        &(RGBColor(128, 0, 128)),   // Purple
        &(RGBColor(255, 192, 203)), // Pink
        &(RGBColor(165, 42, 42)),   // Brown
    ];

    // Draw squares
    for (i, square) in squares.iter().enumerate() {
        let color = colors[i % colors.len()];

        if square.angle == 0.0 {
            // Simple rectangle for axis-aligned squares
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (square.x, square.y),
                    (square.x + square.size, square.y + square.size),
                ],
                color.mix(0.7).filled().stroke_width(2),
            )))?;
        } else {
            // Draw rotated square as polygon
            let corners = square.corners();
            let points: Vec<(f64, f64)> = corners.iter().map(|p| (p.x, p.y)).collect();

            chart.draw_series(std::iter::once(Polygon::new(
                points.clone(),
                color.mix(0.7).filled().stroke_width(2),
            )))?;

            // Draw outline
            for i in 0..4 {
                let start = points[i];
                let end = points[(i + 1) % 4];
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![start, end],
                    BLACK.stroke_width(2),
                )))?;
            }
        }

        // Add square number label
        let center = square.center();
        chart.draw_series(std::iter::once(Text::new(
            format!("{}", i + 1),
            (center.x, center.y),
            ("sans-serif", 20).into_font().color(&BLACK),
        )))?;
    }

    // Add statistics text
    let efficiency = (squares.len() as f64 / container_size.powi(2)) * 100.0;
    let stats_text = format!(
        "Efficiency: {:.2}%\nContainer: {:.4}\nSquares: {}",
        efficiency,
        container_size,
        squares.len()
    );

    chart.draw_series(std::iter::once(Text::new(
        stats_text,
        (container_size * 0.02, container_size * 0.95),
        ("sans-serif", 16).into_font().color(&BLACK),
    )))?;

    root.present()?;
    Ok(())
}

pub fn create_convergence_plot(
    iteration_history: &[crate::solver::IterationRecord],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    if iteration_history.is_empty() {
        return Err("No iteration history to plot".into());
    }

    let root = BitMapBackend::new(filename, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_iter = iteration_history.last().unwrap().iteration as f64;
    let min_size = iteration_history
        .iter()
        .map(|r| r.best_size)
        .fold(f64::INFINITY, f64::min);
    let max_size = iteration_history
        .iter()
        .map(|r| r.container_size)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Convergence History", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_iter, min_size * 0.98..max_size * 1.02)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Container Size")
        .draw()?;

    // Draw current container size
    chart.draw_series(LineSeries::new(
        iteration_history
            .iter()
            .map(|r| (r.iteration as f64, r.container_size)),
        BLUE.stroke_width(2),
    ))?;

    // Draw best size found so far
    chart.draw_series(LineSeries::new(
        iteration_history
            .iter()
            .map(|r| (r.iteration as f64, r.best_size)),
        RED.stroke_width(3),
    ))?;

    // Mark improvements
    chart.draw_series(
        iteration_history
            .iter()
            .filter(|r| r.improvement)
            .map(|r| Circle::new((r.iteration as f64, r.best_size), 5, GREEN.filled())),
    )?;

    // Legend
    chart.draw_series(std::iter::once(Text::new(
        "Blue: Current size\nRed: Best size\nGreen: Improvements",
        (max_iter * 0.7, max_size * 0.95),
        ("sans-serif", 16).into_font(),
    )))?;

    root.present()?;
    Ok(())
}

pub fn create_animation_frames(
    solutions: &[(Vec<Square>, f64)],
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(output_dir)?;

    for (frame, (squares, container_size)) in solutions.iter().enumerate() {
        let filename = format!("{}/frame_{:04}.png", output_dir, frame);
        create_visualization(squares, *container_size, &filename)?;

        if frame % 10 == 0 {
            println!("Generated frame {}/{}", frame + 1, solutions.len());
        }
    }

    println!("Animation frames saved to {}", output_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_creation() {
        let squares = vec![
            Square::new(0.0, 0.0, 1.0, 0.0),
            Square::new(1.5, 0.0, 1.0, 0.0),
            Square::new(0.0, 1.5, 1.0, 0.0),
        ];

        // This test just ensures the function doesn't panic
        // In a real scenario, you'd check if the file was created
        let result = create_visualization(&squares, 3.0, "test_output.png");
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file("test_output.png");
    }
}
