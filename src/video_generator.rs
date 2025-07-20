use crate::animation::*;
use crate::animation_renderer::*;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

pub enum VideoFormat {
    Mp4,
    Gif,
    Webm,
}

impl VideoFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp4" => Some(VideoFormat::Mp4),
            "gif" => Some(VideoFormat::Gif),
            "webm" => Some(VideoFormat::Webm),
            _ => None,
        }
    }

    pub fn codec(&self) -> &'static str {
        match self {
            VideoFormat::Mp4 => "libx264",
            VideoFormat::Gif => "gif",
            VideoFormat::Webm => "libvpx-vp9",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            VideoFormat::Mp4 => "mp4",
            VideoFormat::Gif => "gif",
            VideoFormat::Webm => "webm",
        }
    }

    pub fn pixel_format(&self) -> &'static str {
        match self {
            VideoFormat::Mp4 => "yuv420p",
            VideoFormat::Gif => "pal8",
            VideoFormat::Webm => "yuva420p",
        }
    }
}

pub struct VideoGeneratorConfig {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub format: VideoFormat,
    pub interpolate: bool,
    pub quality: VideoQuality,
}

pub enum VideoQuality {
    Low,    // Fast encoding, larger files
    Medium, // Balanced
    High,   // Slow encoding, smaller files
}

impl VideoQuality {
    pub fn crf_value(&self) -> u32 {
        match self {
            VideoQuality::Low => 28,
            VideoQuality::Medium => 23,
            VideoQuality::High => 18,
        }
    }

    pub fn preset(&self) -> &'static str {
        match self {
            VideoQuality::Low => "ultrafast",
            VideoQuality::Medium => "medium",
            VideoQuality::High => "slow",
        }
    }
}

impl Default for VideoGeneratorConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            fps: 30.0,
            format: VideoFormat::Mp4,
            interpolate: true,
            quality: VideoQuality::Medium,
        }
    }
}

pub struct VideoGenerator {
    config: VideoGeneratorConfig,
}

impl VideoGenerator {
    pub fn new(config: VideoGeneratorConfig) -> Self {
        Self { config }
    }

    pub fn generate_video<P: AsRef<Path>>(
        &self,
        animation_data: &AnimationData,
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Generating video with FFmpeg...");
        
        // Create temporary directory for frames
        let temp_dir = TempDir::new()?;
        let frames_dir = temp_dir.path();
        
        // Step 1: Render frames
        println!("Rendering frames...");
        let renderer = AnimationSequenceRenderer::new(
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.interpolate,
        );
        
        let frame_files = renderer.render_animation_sequence(animation_data, frames_dir)?;
        
        if frame_files.is_empty() {
            return Err("No frames were generated".into());
        }
        
        // Step 2: Generate video using FFmpeg
        println!("Encoding video...");
        self.encode_video(frames_dir, &output_path, frame_files.len())?;
        
        println!("Video generation complete: {}", output_path.as_ref().display());
        Ok(())
    }

    fn encode_video<P: AsRef<Path>>(
        &self,
        frames_dir: &Path,
        output_path: P,
        frame_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let input_pattern = frames_dir.join("frame_%06d.png");
        let output_path = output_path.as_ref();
        
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y") // Overwrite output file
            .arg("-framerate")
            .arg(self.config.fps.to_string())
            .arg("-i")
            .arg(&input_pattern)
            .arg("-frames:v")
            .arg(frame_count.to_string());
        
        // Add format-specific options
        match self.config.format {
            VideoFormat::Mp4 => {
                cmd.arg("-c:v")
                    .arg(self.config.format.codec())
                    .arg("-preset")
                    .arg(self.config.quality.preset())
                    .arg("-crf")
                    .arg(self.config.quality.crf_value().to_string())
                    .arg("-pix_fmt")
                    .arg(self.config.format.pixel_format())
                    .arg("-movflags")
                    .arg("+faststart"); // Enable fast start for web playback
            }
            VideoFormat::Gif => {
                // For GIF, we need a two-pass approach for better quality
                let palette_path = frames_dir.join("palette.png");
                
                // First pass - generate palette
                let mut palette_cmd = Command::new("ffmpeg");
                palette_cmd.arg("-y")
                    .arg("-framerate")
                    .arg(self.config.fps.to_string())
                    .arg("-i")
                    .arg(&input_pattern)
                    .arg("-vf")
                    .arg("fps=15,scale=640:-1:flags=lanczos,palettegen=reserve_transparent=0")
                    .arg(&palette_path);
                
                let palette_output = palette_cmd.output()?;
                if !palette_output.status.success() {
                    return Err(format!("FFmpeg palette generation failed: {}", 
                        String::from_utf8_lossy(&palette_output.stderr)).into());
                }
                
                // Second pass - create GIF with palette
                cmd = Command::new("ffmpeg");
                cmd.arg("-y")
                    .arg("-framerate")
                    .arg(self.config.fps.to_string())
                    .arg("-i")
                    .arg(&input_pattern)
                    .arg("-i")
                    .arg(&palette_path)
                    .arg("-frames:v")
                    .arg(frame_count.to_string())
                    .arg("-lavfi")
                    .arg("fps=15,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse");
            }
            VideoFormat::Webm => {
                cmd.arg("-c:v")
                    .arg(self.config.format.codec())
                    .arg("-b:v")
                    .arg("1M") // 1 Mbps bitrate
                    .arg("-crf")
                    .arg(self.config.quality.crf_value().to_string())
                    .arg("-pix_fmt")
                    .arg(self.config.format.pixel_format());
            }
        }
        
        cmd.arg(output_path);
        
        println!("Running FFmpeg...");
        
        let output = cmd.output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("FFmpeg encoding failed: {}", stderr).into());
        }
        
        Ok(())
    }

    pub fn estimate_output_size(
        &self,
        animation_data: &AnimationData,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let duration = animation_data.metadata.total_duration;
        let frame_count = if self.config.interpolate {
            (duration * self.config.fps).ceil() as usize
        } else {
            animation_data.frames.len()
        };
        
        let estimated_mb = match self.config.format {
            VideoFormat::Mp4 => {
                // Rough estimate: 1-5 MB per minute depending on quality
                let bitrate_factor = match self.config.quality {
                    VideoQuality::Low => 2.0,
                    VideoQuality::Medium => 3.5,
                    VideoQuality::High => 5.0,
                };
                (duration / 60.0) * bitrate_factor
            }
            VideoFormat::Gif => {
                // GIFs are typically much larger
                let pixels_per_frame = (self.config.width * self.config.height) as f64;
                let estimated_bytes = frame_count as f64 * pixels_per_frame * 0.1; // Rough estimate
                estimated_bytes / (1024.0 * 1024.0)
            }
            VideoFormat::Webm => {
                // Similar to MP4 but often slightly smaller
                let bitrate_factor = match self.config.quality {
                    VideoQuality::Low => 1.5,
                    VideoQuality::Medium => 3.0,
                    VideoQuality::High => 4.5,
                };
                (duration / 60.0) * bitrate_factor
            }
        };
        
        Ok(format!(
            "Estimated size: {:.1} MB ({} frames, {:.1}s duration)",
            estimated_mb, frame_count, duration
        ))
    }
}

pub fn detect_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_format_detection() {
        assert!(matches!(VideoFormat::from_extension("mp4"), Some(VideoFormat::Mp4)));
        assert!(matches!(VideoFormat::from_extension("MP4"), Some(VideoFormat::Mp4)));
        assert!(matches!(VideoFormat::from_extension("gif"), Some(VideoFormat::Gif)));
        assert!(matches!(VideoFormat::from_extension("webm"), Some(VideoFormat::Webm)));
        assert!(VideoFormat::from_extension("xyz").is_none());
    }

    #[test]
    fn test_quality_settings() {
        assert_eq!(VideoQuality::Low.crf_value(), 28);
        assert_eq!(VideoQuality::Medium.crf_value(), 23);
        assert_eq!(VideoQuality::High.crf_value(), 18);
    }
}