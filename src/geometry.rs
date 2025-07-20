use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    
    pub fn distance_to(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
    
    pub fn rotate_around(&self, center: &Point, angle: f64) -> Point {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let dx = self.x - center.x;
        let dy = self.y - center.y;
        
        Point::new(
            center.x + dx * cos_a - dy * sin_a,
            center.y + dx * sin_a + dy * cos_a,
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Square {
    pub x: f64,
    pub y: f64,
    pub size: f64,
    pub angle: f64,
}

impl Square {
    pub fn new(x: f64, y: f64, size: f64, angle: f64) -> Self {
        Self { x, y, size, angle }
    }
    
    pub fn center(&self) -> Point {
        Point::new(self.x + self.size / 2.0, self.y + self.size / 2.0)
    }
    
    pub fn corners(&self) -> [Point; 4] {
        let center = self.center();
        let half_size = self.size / 2.0;
        
        let corners = [
            Point::new(-half_size, -half_size),
            Point::new(half_size, -half_size),
            Point::new(half_size, half_size),
            Point::new(-half_size, half_size),
        ];
        
        corners.map(|p| p.rotate_around(&Point::new(0.0, 0.0), self.angle))
               .map(|p| Point::new(p.x + center.x, p.y + center.y))
    }
    
    pub fn overlaps_with(&self, other: &Square) -> bool {
        if self.angle == 0.0 && other.angle == 0.0 {
            // Fast AABB check for axis-aligned squares
            self.x < other.x + other.size &&
            self.x + self.size > other.x &&
            self.y < other.y + other.size &&
            self.y + self.size > other.y
        } else {
            // SAT (Separating Axis Theorem) for rotated squares
            self.sat_overlap(other)
        }
    }
    
    fn sat_overlap(&self, other: &Square) -> bool {
        let corners1 = self.corners();
        let corners2 = other.corners();
        
        // Test all edge normals from both squares
        for corners in [&corners1, &corners2] {
            for i in 0..4 {
                let edge = Point::new(
                    corners[(i + 1) % 4].x - corners[i].x,
                    corners[(i + 1) % 4].y - corners[i].y,
                );
                
                // Normal to the edge (perpendicular)
                let normal = Point::new(-edge.y, edge.x);
                let length = (normal.x * normal.x + normal.y * normal.y).sqrt();
                
                if length < 1e-10 {
                    continue;
                }
                
                let normal = Point::new(normal.x / length, normal.y / length);
                
                // Project both squares onto this normal
                let proj1: Vec<f64> = corners1.iter()
                    .map(|c| normal.x * c.x + normal.y * c.y)
                    .collect();
                let proj2: Vec<f64> = corners2.iter()
                    .map(|c| normal.x * c.x + normal.y * c.y)
                    .collect();
                
                let min1 = proj1.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max1 = proj1.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min2 = proj2.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max2 = proj2.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                // Check for separation
                if max1 < min2 || max2 < min1 {
                    return false; // Separating axis found
                }
            }
        }
        
        true // No separating axis found, squares overlap
    }
    
    pub fn is_inside_container(&self, container_size: f64) -> bool {
        if self.angle == 0.0 {
            // Fast check for axis-aligned squares
            self.x >= 0.0 && self.y >= 0.0 &&
            self.x + self.size <= container_size &&
            self.y + self.size <= container_size
        } else {
            // Check all corners for rotated squares
            self.corners().iter().all(|corner| {
                corner.x >= 0.0 && corner.x <= container_size &&
                corner.y >= 0.0 && corner.y <= container_size
            })
        }
    }
    
    pub fn bounding_box(&self) -> (Point, Point) {
        if self.angle == 0.0 {
            (
                Point::new(self.x, self.y),
                Point::new(self.x + self.size, self.y + self.size),
            )
        } else {
            let corners = self.corners();
            let min_x = corners.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
            let max_x = corners.iter().map(|c| c.x).fold(f64::NEG_INFINITY, f64::max);
            let min_y = corners.iter().map(|c| c.y).fold(f64::INFINITY, f64::min);
            let max_y = corners.iter().map(|c| c.y).fold(f64::NEG_INFINITY, f64::max);
            
            (Point::new(min_x, min_y), Point::new(max_x, max_y))
        }
    }
    
    pub fn area(&self) -> f64 {
        self.size * self.size
    }
    
    pub fn translate(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
    
    pub fn rotate(&mut self, angle: f64) {
        self.angle = (self.angle + angle) % (PI / 2.0);
    }
    
    pub fn set_position(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }
}

pub struct Container {
    pub size: f64,
}

impl Container {
    pub fn new(size: f64) -> Self {
        Self { size }
    }
    
    pub fn area(&self) -> f64 {
        self.size * self.size
    }
    
    pub fn contains_square(&self, square: &Square) -> bool {
        square.is_inside_container(self.size)
    }
    
    pub fn efficiency(&self, squares: &[Square]) -> f64 {
        let used_area: f64 = squares.iter().map(|s| s.area()).sum();
        used_area / self.area()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point_rotation() {
        let p = Point::new(1.0, 0.0);
        let center = Point::new(0.0, 0.0);
        let rotated = p.rotate_around(&center, PI / 2.0);
        
        assert!((rotated.x - 0.0).abs() < 1e-10);
        assert!((rotated.y - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_square_overlap_axis_aligned() {
        let sq1 = Square::new(0.0, 0.0, 1.0, 0.0);
        let sq2 = Square::new(0.5, 0.5, 1.0, 0.0);
        let sq3 = Square::new(2.0, 2.0, 1.0, 0.0);
        
        assert!(sq1.overlaps_with(&sq2));
        assert!(!sq1.overlaps_with(&sq3));
    }
    
    #[test]
    fn test_square_container_containment() {
        let container = Container::new(10.0);
        let sq1 = Square::new(1.0, 1.0, 2.0, 0.0);
        let sq2 = Square::new(9.0, 9.0, 2.0, 0.0); // Goes outside
        
        assert!(container.contains_square(&sq1));
        assert!(!container.contains_square(&sq2));
    }
    
    #[test]
    fn test_rotated_square_corners() {
        let sq = Square::new(0.0, 0.0, 2.0, PI / 4.0);
        let corners = sq.corners();
        
        // For a 2x2 square rotated 45 degrees, corners should be at specific positions
        // This is a basic sanity check that rotation is working
        assert_eq!(corners.len(), 4);
        
        // Check that corners form a valid square (all distances from center are equal)
        let center = sq.center();
        let distances: Vec<f64> = corners.iter()
            .map(|c| center.distance_to(c))
            .collect();
        
        let expected_distance = 2.0_f64.sqrt(); // sqrt(2) for a 2x2 square
        for &dist in &distances {
            assert!((dist - expected_distance).abs() < 1e-10);
        }
    }
}