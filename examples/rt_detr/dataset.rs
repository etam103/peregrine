use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Annotation {
    pub class_id: usize,
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub image_path: PathBuf,
    pub annotations: Vec<Annotation>,
}

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

    fn load_image(&self, idx: usize) -> Vec<f32>;
    fn get_targets(&self, idx: usize) -> Vec<Target>;

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

fn parse_voc_xml(path: &Path, class_to_id: &HashMap<String, usize>) -> (String, Vec<Annotation>) {
    let xml_content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    let mut reader = Reader::from_str(&xml_content);

    let mut filename = String::new();
    let mut annotations = Vec::new();

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
// COCO JSON dataset
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CocoJson {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct CocoAnnotation {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
}

#[derive(Deserialize)]
struct CocoCategory {
    id: u64,
    name: String,
}

pub struct CocoDataset {
    samples: Vec<Sample>,
    classes: Vec<String>,
    image_size: usize,
}

impl CocoDataset {
    pub fn load(json_path: &str, image_dir: &str, image_size: usize, class_names: &[&str]) -> Self {
        let classes: Vec<String> = class_names.iter().map(|s| s.to_string()).collect();
        let class_set: HashMap<&str, usize> = class_names
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect();

        let json_content = fs::read_to_string(json_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {}", json_path, e));
        let coco: CocoJson = serde_json::from_str(&json_content)
            .unwrap_or_else(|e| panic!("cannot parse {}: {}", json_path, e));

        let cat_to_class: HashMap<u64, usize> = coco.categories.iter()
            .filter_map(|cat| class_set.get(cat.name.as_str()).map(|&id| (cat.id, id)))
            .collect();

        let image_map: HashMap<u64, &CocoImage> = coco.images.iter()
            .map(|img| (img.id, img))
            .collect();

        let mut anns_per_image: HashMap<u64, Vec<Annotation>> = HashMap::new();
        for ann in &coco.annotations {
            if let Some(&class_id) = cat_to_class.get(&ann.category_id) {
                let [x, y, w, h] = ann.bbox;
                anns_per_image.entry(ann.image_id).or_default().push(Annotation {
                    class_id,
                    xmin: x,
                    ymin: y,
                    xmax: x + w,
                    ymax: y + h,
                });
            }
        }

        let image_dir = PathBuf::from(image_dir);
        let mut samples: Vec<Sample> = anns_per_image.into_iter()
            .filter_map(|(image_id, annotations)| {
                let img = image_map.get(&image_id)?;
                let image_path = image_dir.join(&img.file_name);
                if !image_path.exists() {
                    return None;
                }
                Some(Sample { image_path, annotations })
            })
            .collect();

        samples.sort_by(|a, b| a.image_path.cmp(&b.image_path));

        eprintln!(
            "CocoDataset: loaded {} samples with {} classes from {}",
            samples.len(), classes.len(), json_path,
        );

        CocoDataset { samples, classes, image_size }
    }
}

impl Dataset for CocoDataset {
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

        sample.annotations.iter().map(|ann| {
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
        }).collect()
    }
}
