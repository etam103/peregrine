use std::collections::HashMap;
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use quick_xml::events::Event;
use quick_xml::Reader;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single bounding-box annotation: class index + pixel-space coords.
#[derive(Clone, Debug)]
pub struct Annotation {
    pub class_id: usize,
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
}

/// One sample: path to the image file and its associated annotations.
#[derive(Clone, Debug)]
pub struct Sample {
    pub image_path: PathBuf,
    pub annotations: Vec<Annotation>,
}

/// A normalized target entry in YOLO-style format, ready for loss computation.
/// All coordinates are relative to the image dimensions (0..1).
#[derive(Clone, Debug)]
pub struct Target {
    pub batch_idx: usize,
    pub class_id: usize,
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

// ---------------------------------------------------------------------------
// Dataset trait
// ---------------------------------------------------------------------------

pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn image_size(&self) -> usize;
    fn num_classes(&self) -> usize;
    fn class_name(&self, id: usize) -> &str;

    /// Load and preprocess a single image, returning CHW f32 data in [0, 1].
    fn load_image(&self, idx: usize) -> Vec<f32>;

    /// Return annotations for sample `idx` normalized to [0, 1].
    fn get_targets(&self, idx: usize) -> Vec<Target>;

    /// Yield a batch of `(image_data, targets)`.
    ///
    /// `image_data` is a flat `Vec<f32>` of shape `[batch, 3, H, W]`.
    /// `targets` carries per-object entries with `batch_idx` set relative to
    /// this batch.
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Vec<f32>, Vec<Target>) {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(self.len());
        let sz = self.image_size();
        let pixels_per_image = 3 * sz * sz;

        let mut images = Vec::with_capacity((end - start) * pixels_per_image);
        let mut targets = Vec::new();

        for (local_idx, global_idx) in (start..end).enumerate() {
            images.extend_from_slice(&self.load_image(global_idx));
            for mut t in self.get_targets(global_idx) {
                t.batch_idx = local_idx;
                targets.push(t);
            }
        }
        (images, targets)
    }

    fn num_batches(&self, batch_size: usize) -> usize {
        (self.len() + batch_size - 1) / batch_size
    }
}

// ---------------------------------------------------------------------------
// Image loading helpers
// ---------------------------------------------------------------------------

/// Load an image from disk, resize to `size x size`, normalize to [0,1],
/// and return as a flat `Vec<f32>` in CHW layout (3 channels).
fn load_and_preprocess(path: &Path, size: usize) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open image {}: {}", path.display(), e));
    let img = img.resize_exact(size as u32, size as u32, FilterType::Triangle);
    let rgb = img.to_rgb8();

    let mut chw = vec![0.0f32; 3 * size * size];
    for y in 0..size {
        for x in 0..size {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let offset = y * size + x;
            chw[0 * size * size + offset] = pixel[0] as f32 / 255.0;
            chw[1 * size * size + offset] = pixel[1] as f32 / 255.0;
            chw[2 * size * size + offset] = pixel[2] as f32 / 255.0;
        }
    }
    chw
}

// ---------------------------------------------------------------------------
// Pascal VOC dataset
// ---------------------------------------------------------------------------

pub struct VocDataset {
    samples: Vec<Sample>,
    classes: Vec<String>,
    image_size: usize,
}

impl VocDataset {
    /// Load a Pascal VOC dataset from a root directory (e.g.
    /// `VOCdevkit/VOC2012`).
    ///
    /// `root` must contain `Annotations/` and `JPEGImages/` subdirectories.
    /// Only objects whose class name appears in `class_names` are kept.
    /// Images that contain zero matching objects are skipped.
    pub fn load(root: &str, image_size: usize, class_names: &[&str]) -> Self {
        let root = PathBuf::from(root);
        let ann_dir = root.join("Annotations");
        let img_dir = root.join("JPEGImages");

        assert!(ann_dir.is_dir(), "Annotations directory not found: {}", ann_dir.display());
        assert!(img_dir.is_dir(), "JPEGImages directory not found: {}", img_dir.display());

        let class_to_id: HashMap<String, usize> = class_names
            .iter()
            .enumerate()
            .map(|(i, &name)| (name.to_string(), i))
            .collect();
        let classes: Vec<String> = class_names.iter().map(|s| s.to_string()).collect();

        let mut xml_paths: Vec<_> = fs::read_dir(&ann_dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {}", ann_dir.display(), e))
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("xml") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        xml_paths.sort();

        let mut samples = Vec::new();
        for xml_path in &xml_paths {
            let (filename, annotations) = parse_voc_xml(xml_path, &class_to_id);
            if annotations.is_empty() {
                continue;
            }
            let image_path = img_dir.join(&filename);
            if !image_path.exists() {
                eprintln!("warning: image not found, skipping: {}", image_path.display());
                continue;
            }
            samples.push(Sample { image_path, annotations });
        }

        eprintln!(
            "VocDataset: loaded {} samples with {} classes from {}",
            samples.len(),
            classes.len(),
            root.display()
        );

        VocDataset { samples, classes, image_size }
    }
}

/// Parse a single VOC XML annotation file and return (image filename, annotations).
/// Only objects whose class is present in `class_to_id` are returned.
fn parse_voc_xml(path: &Path, class_to_id: &HashMap<String, usize>) -> (String, Vec<Annotation>) {
    let xml_content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    let mut reader = Reader::from_str(&xml_content);

    let mut filename = String::new();
    let mut annotations = Vec::new();

    // State machine for walking <object> elements.
    let mut current_tag = String::new();
    let mut in_object = false;
    let mut in_bndbox = false;
    let mut obj_name = String::new();
    let mut xmin: f32 = 0.0;
    let mut ymin: f32 = 0.0;
    let mut xmax: f32 = 0.0;
    let mut ymax: f32 = 0.0;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                current_tag = tag.clone();
                match tag.as_str() {
                    "object" => {
                        in_object = true;
                        obj_name.clear();
                        xmin = 0.0;
                        ymin = 0.0;
                        xmax = 0.0;
                        ymax = 0.0;
                    }
                    "bndbox" if in_object => {
                        in_bndbox = true;
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "object" => {
                        if let Some(&class_id) = class_to_id.get(&obj_name) {
                            annotations.push(Annotation { class_id, xmin, ymin, xmax, ymax });
                        }
                        in_object = false;
                        in_bndbox = false;
                    }
                    "bndbox" => {
                        in_bndbox = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().trim().to_string();
                if text.is_empty() {
                    continue;
                }
                if !in_object {
                    if current_tag == "filename" {
                        filename = text;
                    }
                } else if in_bndbox {
                    match current_tag.as_str() {
                        "xmin" => xmin = text.parse().unwrap_or(0.0),
                        "ymin" => ymin = text.parse().unwrap_or(0.0),
                        "xmax" => xmax = text.parse().unwrap_or(0.0),
                        "ymax" => ymax = text.parse().unwrap_or(0.0),
                        _ => {}
                    }
                } else if current_tag == "name" {
                    obj_name = text;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => panic!("XML parse error in {}: {}", path.display(), e),
            _ => {}
        }
    }

    (filename, annotations)
}

impl Dataset for VocDataset {
    fn len(&self) -> usize { self.samples.len() }
    fn image_size(&self) -> usize { self.image_size }
    fn num_classes(&self) -> usize { self.classes.len() }
    fn class_name(&self, id: usize) -> &str { &self.classes[id] }

    fn load_image(&self, idx: usize) -> Vec<f32> {
        load_and_preprocess(&self.samples[idx].image_path, self.image_size)
    }

    fn get_targets(&self, idx: usize) -> Vec<Target> {
        let sample = &self.samples[idx];
        // We need the original image dimensions to normalize bbox coords.
        let (orig_w, orig_h) = image::image_dimensions(&sample.image_path)
            .unwrap_or_else(|e| {
                panic!("cannot read dimensions of {}: {}", sample.image_path.display(), e)
            });
        let ow = orig_w as f32;
        let oh = orig_h as f32;

        sample
            .annotations
            .iter()
            .map(|ann| {
                let cx = ((ann.xmin + ann.xmax) / 2.0) / ow;
                let cy = ((ann.ymin + ann.ymax) / 2.0) / oh;
                let w = (ann.xmax - ann.xmin) / ow;
                let h = (ann.ymax - ann.ymin) / oh;
                Target {
                    batch_idx: 0, // filled in by get_batch
                    class_id: ann.class_id,
                    cx: cx.clamp(0.0, 1.0),
                    cy: cy.clamp(0.0, 1.0),
                    w: w.clamp(0.0, 1.0),
                    h: h.clamp(0.0, 1.0),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CSV dataset (simpler fallback)
// ---------------------------------------------------------------------------

/// A simple CSV-based object detection dataset.
///
/// Expected format (one row per object, with a header row):
/// ```text
/// image_path,class,xmin,ymin,xmax,ymax
/// images/001.jpg,car,100,50,300,200
/// images/001.jpg,person,350,60,400,180
/// ```
///
/// Pixel coordinates are in the original image space and will be normalized
/// using the actual image dimensions when loading.
pub struct CsvDataset {
    samples: Vec<Sample>,
    classes: Vec<String>,
    image_size: usize,
}

impl CsvDataset {
    /// Parse a CSV file and build the dataset.
    ///
    /// `csv_path` is the path to the CSV file. Image paths inside the CSV are
    /// resolved relative to `image_root`. Only classes listed in
    /// `class_names` are kept; images with zero matching objects are dropped.
    pub fn load(csv_path: &str, image_root: &str, image_size: usize, class_names: &[&str]) -> Self {
        let class_to_id: HashMap<String, usize> = class_names
            .iter()
            .enumerate()
            .map(|(i, &name)| (name.to_string(), i))
            .collect();
        let classes: Vec<String> = class_names.iter().map(|s| s.to_string()).collect();
        let image_root = PathBuf::from(image_root);

        // Group annotations by image path.
        let mut per_image: HashMap<PathBuf, Vec<Annotation>> = HashMap::new();

        let file = fs::File::open(csv_path)
            .unwrap_or_else(|e| panic!("cannot open CSV {}: {}", csv_path, e));
        let reader = std::io::BufReader::new(file);

        for (line_no, line) in reader.lines().enumerate() {
            let line = line.unwrap_or_else(|e| panic!("read error at line {}: {}", line_no, e));
            let line = line.trim().to_string();
            if line.is_empty() || line_no == 0 {
                continue; // skip header
            }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 6 {
                eprintln!("warning: skipping malformed CSV line {}: {}", line_no + 1, line);
                continue;
            }
            let class_name = cols[1].trim();
            let class_id = match class_to_id.get(class_name) {
                Some(&id) => id,
                None => continue,
            };
            let xmin: f32 = cols[2].trim().parse().unwrap_or(0.0);
            let ymin: f32 = cols[3].trim().parse().unwrap_or(0.0);
            let xmax: f32 = cols[4].trim().parse().unwrap_or(0.0);
            let ymax: f32 = cols[5].trim().parse().unwrap_or(0.0);

            let image_path = image_root.join(cols[0].trim());
            per_image
                .entry(image_path)
                .or_default()
                .push(Annotation { class_id, xmin, ymin, xmax, ymax });
        }

        let mut samples: Vec<Sample> = per_image
            .into_iter()
            .filter(|(path, _)| {
                if !path.exists() {
                    eprintln!("warning: image not found, skipping: {}", path.display());
                    return false;
                }
                true
            })
            .map(|(image_path, annotations)| Sample { image_path, annotations })
            .collect();

        // Stable ordering for reproducibility.
        samples.sort_by(|a, b| a.image_path.cmp(&b.image_path));

        eprintln!(
            "CsvDataset: loaded {} samples with {} classes from {}",
            samples.len(),
            classes.len(),
            csv_path
        );

        CsvDataset { samples, classes, image_size }
    }
}

impl Dataset for CsvDataset {
    fn len(&self) -> usize { self.samples.len() }
    fn image_size(&self) -> usize { self.image_size }
    fn num_classes(&self) -> usize { self.classes.len() }
    fn class_name(&self, id: usize) -> &str { &self.classes[id] }

    fn load_image(&self, idx: usize) -> Vec<f32> {
        load_and_preprocess(&self.samples[idx].image_path, self.image_size)
    }

    fn get_targets(&self, idx: usize) -> Vec<Target> {
        let sample = &self.samples[idx];
        let (orig_w, orig_h) = image::image_dimensions(&sample.image_path)
            .unwrap_or_else(|e| {
                panic!("cannot read dimensions of {}: {}", sample.image_path.display(), e)
            });
        let ow = orig_w as f32;
        let oh = orig_h as f32;

        sample
            .annotations
            .iter()
            .map(|ann| {
                let cx = ((ann.xmin + ann.xmax) / 2.0) / ow;
                let cy = ((ann.ymin + ann.ymax) / 2.0) / oh;
                let w = (ann.xmax - ann.xmin) / ow;
                let h = (ann.ymax - ann.ymin) / oh;
                Target {
                    batch_idx: 0,
                    class_id: ann.class_id,
                    cx: cx.clamp(0.0, 1.0),
                    cy: cy.clamp(0.0, 1.0),
                    w: w.clamp(0.0, 1.0),
                    h: h.clamp(0.0, 1.0),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Convenience: flatten targets into a dense [N, 6] Vec<f32>
// ---------------------------------------------------------------------------

/// Convert a `Vec<Target>` into a flat `Vec<f32>` where each target is
/// encoded as `[batch_idx, class_id, cx, cy, w, h]`.
pub fn targets_to_flat(targets: &[Target]) -> Vec<f32> {
    let mut out = Vec::with_capacity(targets.len() * 6);
    for t in targets {
        out.push(t.batch_idx as f32);
        out.push(t.class_id as f32);
        out.push(t.cx);
        out.push(t.cy);
        out.push(t.w);
        out.push(t.h);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Build a tiny VOC-format dataset on disk and verify round-trip loading.
    #[test]
    fn test_voc_loading() {
        let tmp = std::env::temp_dir().join("rustorch_voc_test");
        let _ = fs::remove_dir_all(&tmp);
        let ann_dir = tmp.join("Annotations");
        let img_dir = tmp.join("JPEGImages");
        fs::create_dir_all(&ann_dir).unwrap();
        fs::create_dir_all(&img_dir).unwrap();

        // Write a small 4x4 red JPEG.
        let mut img = image::RgbImage::new(4, 4);
        for p in img.pixels_mut() {
            *p = image::Rgb([255, 0, 0]);
        }
        img.save(img_dir.join("test.jpg")).unwrap();

        // Write matching XML annotation.
        let xml = r#"<annotation>
  <filename>test.jpg</filename>
  <object>
    <name>car</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>3</xmax><ymax>3</ymax></bndbox>
  </object>
  <object>
    <name>cat</name>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox>
  </object>
</annotation>"#;
        fs::write(ann_dir.join("test.xml"), xml).unwrap();

        // Only request "car" — "cat" should be filtered out.
        let ds = VocDataset::load(tmp.to_str().unwrap(), 8, &["car"]);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.num_classes(), 1);

        let targets = ds.get_targets(0);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].class_id, 0);

        let img_data = ds.load_image(0);
        assert_eq!(img_data.len(), 3 * 8 * 8);
        // All values should be in [0, 1].
        assert!(img_data.iter().all(|&v| (0.0..=1.0).contains(&v)));

        let (batch_imgs, batch_targets) = ds.get_batch(0, 1);
        assert_eq!(batch_imgs.len(), 3 * 8 * 8);
        assert_eq!(batch_targets.len(), 1);
        assert_eq!(batch_targets[0].batch_idx, 0);

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Build a tiny CSV dataset on disk and verify round-trip loading.
    #[test]
    fn test_csv_loading() {
        let tmp = std::env::temp_dir().join("rustorch_csv_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        // Write a 4x4 blue image.
        let mut img = image::RgbImage::new(4, 4);
        for p in img.pixels_mut() {
            *p = image::Rgb([0, 0, 255]);
        }
        img.save(tmp.join("blue.png")).unwrap();

        let csv = "image_path,class,xmin,ymin,xmax,ymax\nblue.png,dog,0,0,4,4\nblue.png,person,1,1,3,3\n";
        let csv_path = tmp.join("labels.csv");
        fs::write(&csv_path, csv).unwrap();

        let ds = CsvDataset::load(
            csv_path.to_str().unwrap(),
            tmp.to_str().unwrap(),
            16,
            &["dog", "person"],
        );
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.num_classes(), 2);

        let targets = ds.get_targets(0);
        assert_eq!(targets.len(), 2);

        let flat = targets_to_flat(&targets);
        assert_eq!(flat.len(), 12); // 2 targets * 6 fields

        let _ = fs::remove_dir_all(&tmp);
    }
}
