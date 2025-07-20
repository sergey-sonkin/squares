use std::path::{Path, PathBuf};
use std::fs;

pub struct OutputManager {
    base_dir: PathBuf,
}

impl OutputManager {
    pub fn new() -> Self {
        let base_dir = PathBuf::from("outputs");
        Self { base_dir }
    }

    pub fn with_base_dir<P: AsRef<Path>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// Ensure all output directories exist
    pub fn create_directories(&self) -> std::io::Result<()> {
        let dirs = ["json", "mp4", "gif", "webm", "frames", "images"];
        
        for dir in &dirs {
            let path = self.base_dir.join(dir);
            fs::create_dir_all(&path)?;
        }
        
        Ok(())
    }

    /// Get the appropriate output path for a given file
    pub fn get_output_path(&self, filename: &str) -> PathBuf {
        let extension = Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        let subdir = match extension.to_lowercase().as_str() {
            "json" => "json",
            "mp4" => "mp4", 
            "gif" => "gif",
            "webm" => "webm",
            "png" | "jpg" | "jpeg" => "images",
            _ => {
                // Check if it's a frames directory
                if filename.contains("frames") || filename.ends_with("_frames") {
                    "frames"
                } else {
                    "images" // Default fallback
                }
            }
        };

        self.base_dir.join(subdir).join(filename)
    }

    /// Get path for animation JSON files
    pub fn animation_json_path(&self, name: &str) -> PathBuf {
        let filename = if name.ends_with(".json") {
            name.to_string()
        } else {
            format!("{}.json", name)
        };
        self.base_dir.join("json").join(filename)
    }

    /// Get path for video files
    pub fn video_path(&self, name: &str, format: &str) -> PathBuf {
        let filename = if name.contains('.') {
            name.to_string()
        } else {
            format!("{}.{}", name, format)
        };
        
        let subdir = match format.to_lowercase().as_str() {
            "mp4" => "mp4",
            "gif" => "gif", 
            "webm" => "webm",
            _ => "mp4", // Default to mp4
        };
        
        self.base_dir.join(subdir).join(filename)
    }

    /// Get path for frame directories
    pub fn frames_dir_path(&self, name: &str) -> PathBuf {
        let dirname = if name.ends_with("_frames") {
            name.to_string()
        } else {
            format!("{}_frames", name)
        };
        self.base_dir.join("frames").join(dirname)
    }

    /// Get path for static images (visualizations)
    pub fn image_path(&self, name: &str) -> PathBuf {
        let filename = if name.contains('.') {
            name.to_string()
        } else {
            format!("{}.png", name)
        };
        self.base_dir.join("images").join(filename)
    }

    /// Generate a unique filename if the file already exists
    pub fn unique_path(&self, base_path: &Path) -> PathBuf {
        if !base_path.exists() {
            return base_path.to_path_buf();
        }

        let parent = base_path.parent().unwrap_or(Path::new("."));
        let stem = base_path.file_stem().and_then(|s| s.to_str()).unwrap_or("file");
        let extension = base_path.extension().and_then(|s| s.to_str()).unwrap_or("");

        for i in 1..1000 {
            let new_filename = if extension.is_empty() {
                format!("{}_{}", stem, i)
            } else {
                format!("{}_{}.{}", stem, i, extension)
            };
            
            let new_path = parent.join(new_filename);
            if !new_path.exists() {
                return new_path;
            }
        }

        // Fallback with timestamp if we can't find a unique name
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let new_filename = if extension.is_empty() {
            format!("{}_{}", stem, timestamp)
        } else {
            format!("{}_{}.{}", stem, timestamp, extension)
        };
        
        parent.join(new_filename)
    }

    /// Clean up old files (keep only the most recent N files of each type)
    pub fn cleanup_old_files(&self, keep_recent: usize) -> std::io::Result<()> {
        let dirs = ["json", "mp4", "gif", "webm", "images"];
        
        for dir in &dirs {
            let dir_path = self.base_dir.join(dir);
            if !dir_path.exists() {
                continue;
            }

            let mut files: Vec<_> = fs::read_dir(&dir_path)?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let metadata = entry.metadata().ok()?;
                    if metadata.is_file() {
                        Some((entry.path(), metadata.modified().ok()?))
                    } else {
                        None
                    }
                })
                .collect();

            // Sort by modification time (newest first)
            files.sort_by(|a, b| b.1.cmp(&a.1));

            // Remove old files
            for (path, _) in files.into_iter().skip(keep_recent) {
                if let Err(e) = fs::remove_file(&path) {
                    eprintln!("Warning: Failed to remove old file {:?}: {}", path, e);
                }
            }
        }

        Ok(())
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

impl Default for OutputManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_output_path_detection() {
        let temp_dir = TempDir::new().unwrap();
        let manager = OutputManager::with_base_dir(temp_dir.path());

        assert_eq!(
            manager.get_output_path("test.json").file_name().unwrap(),
            "test.json"
        );
        assert!(manager.get_output_path("test.json").parent().unwrap().ends_with("json"));

        assert_eq!(
            manager.get_output_path("video.mp4").file_name().unwrap(),
            "video.mp4"
        );
        assert!(manager.get_output_path("video.mp4").parent().unwrap().ends_with("mp4"));

        assert_eq!(
            manager.get_output_path("anim.gif").file_name().unwrap(),
            "anim.gif"
        );
        assert!(manager.get_output_path("anim.gif").parent().unwrap().ends_with("gif"));
    }

    #[test]
    fn test_specialized_paths() {
        let temp_dir = TempDir::new().unwrap();
        let manager = OutputManager::with_base_dir(temp_dir.path());

        let json_path = manager.animation_json_path("test_animation");
        assert_eq!(json_path.file_name().unwrap(), "test_animation.json");
        assert!(json_path.parent().unwrap().ends_with("json"));

        let video_path = manager.video_path("demo", "mp4");
        assert_eq!(video_path.file_name().unwrap(), "demo.mp4");
        assert!(video_path.parent().unwrap().ends_with("mp4"));

        let frames_path = manager.frames_dir_path("test");
        assert_eq!(frames_path.file_name().unwrap(), "test_frames");
        assert!(frames_path.parent().unwrap().ends_with("frames"));
    }
}