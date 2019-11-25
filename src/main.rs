// The pre-trained weights can be downloaded here:
//   https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/yolo-v3.ot

#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate failure;
extern crate tch;

mod coco_classes;
mod darknet;

use crate::darknet::Darknet;
use image::{imageops, ColorType};
use imageproc::geometric_transformations::Interpolation;
use rocket::State;
use rocket_contrib::json::Json;
use serde::Serialize;
use std::fs::File;
use std::io::{Cursor, Read};
use std::sync::{Arc, RwLock};
use std::time::{Duration};
use tch::nn::{ModuleT};
use tch::vision::image as tch_image;
use tch::Tensor;

#[macro_use]
extern crate rocket;
const CONFIG_NAME: &'static str = "yolo-v3.cfg";
const CONFIDENCE_THRESHOLD: f64 = 0.5;
const NMS_THRESHOLD: f64 = 0.4;

#[derive(Debug, Clone, Copy)]
struct Bbox {
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    confidence: f64,
    class_index: usize,
    class_confidence: f64,
}

// Intersection over union of two bounding boxes.
fn iou(b1: &Bbox, b2: &Bbox) -> f64 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

// Assumes x1 <= x2 and y1 <= y2
pub fn draw_rect(t: &mut Tensor, x1: i64, x2: i64, y1: i64, y2: i64) {
    let color = Tensor::of_slice(&[0., 0., 1.]).view([3, 1, 1]);

    t.narrow(2, x1, x2 - x1)
        .narrow(1, y1, y2 - y1)
        .copy_(&color)
}

// Assumes x1 and y1
pub fn draw_at_x_y(t: &mut Tensor, x: i64, y: i64) {
    let color = Tensor::of_slice(&[1., 0., 1.]).view([3, 1, 1]);

    t.narrow(2, x, 1)
        .narrow(1, y, 1)
        .copy_(&color)
}

pub fn report(pred: &Tensor, img: &Tensor, w: i64, h: i64) -> failure::Fallible<(Tensor, u32)> {
    let (npreds, pred_size) = pred.size2()?;
    let nclasses = (pred_size - 5) as usize;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f64>::from(pred.get(index));
        let confidence = pred[4];
        if confidence > CONFIDENCE_THRESHOLD {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[5 + i] > pred[5 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 5] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    class_index,
                    class_confidence: pred[5 + class_index],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > NMS_THRESHOLD {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
    // Annotate the original image and print boxes information.
    let (_, initial_h, initial_w) = img.size3()?;
    let mut img = img.to_kind(tch::Kind::Float) / 255.;
    let (channels, x_dim, y_dim) = img.size3().expect("Image was not 3 dim tensor");
    println!("ch: {:?}, x: {:?}, y: {:?}", channels, x_dim, y_dim);

    // remember that y starts at top, top is 0
    let max_y_for_given_x_to_consider_people_in_line = |x|{
        let max_y = y_dim-1;
        let initial_y = 100;
        max_y - initial_y - (x as f64/3.2) as i64
    };

    // drawing threshold line for where to consider people in the line or in the street
//    for x in 0..=x_dim-1{
//        for width in 0..=3{
//            draw_at_x_y(&mut img, x, max_y_for_given_x_to_consider_people_in_line(x) + width);
//        }
//    }

    let w_ratio = initial_w as f64 / w as f64;
    let h_ratio = initial_h as f64 / h as f64;
    let mut nb_people = 0;
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        if class_index != 0 {
            continue;
        }
        for b in bboxes_for_class.iter() {
            let xmin = ((b.xmin * w_ratio) as i64).max(0).min(initial_w - 1);
            let ymin = ((b.ymin * h_ratio) as i64).max(0).min(initial_h - 1);
            let xmax = ((b.xmax * w_ratio) as i64).max(0).min(initial_w - 1);
            let ymax = ((b.ymax * h_ratio) as i64).max(0).min(initial_h - 1);
            if ymax >= max_y_for_given_x_to_consider_people_in_line(xmax){
                continue;
            }
            nb_people += 1;
            println!("{}: {:?}", coco_classes::NAMES[class_index], b);
            draw_rect(&mut img, xmin, xmax, ymin, ymax.min(ymin + 2));
            draw_rect(&mut img, xmin, xmax, ymin.max(ymax - 2), ymax);
            draw_rect(&mut img, xmin, xmax.min(xmin + 2), ymin, ymax);
            draw_rect(&mut img, xmin.max(xmax - 2), xmax, ymin, ymax);
        }
    }
    Ok(((img * 255.).to_kind(tch::Kind::Uint8), nb_people))
}

pub struct Yolo {
    darknet: Darknet,
}

impl Yolo {
    pub fn new() -> failure::Fallible<Yolo> {
        // Create the model and load the weights from the file.

        let darknet = darknet::parse_config(CONFIG_NAME)?;

        Ok(Yolo { darknet })
    }

    pub fn process_img(&self) -> failure::Fallible<u32> {
        // Load the image file and resize it.

        let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let model = self.darknet.build_model(&vs.root())?;
        vs.load("yolo-v3.ot")?;
        let net_width = self.darknet.width()?;
        let net_height = self.darknet.height()?;

        let original_image = tch_image::load("rotated_and_cropped_img.png")?;
        let image = tch_image::resize(&original_image, net_width, net_height)?;
        let image = image.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;

        let predictions = model.forward_t(&image, false).squeeze();
        let (image, nb_people) = report(&predictions, &original_image, net_width, net_height)?;
        tch_image::save(&image, format!("processed_img.jpg"))?;
        println!("Converted");
        Ok(nb_people)
    }
}
use rocket::response::status::Custom;
use rocket::http::Status;

#[get("/")]
fn index(
    cached_yolo: State<Arc<RwLock<Yolo>>>,
) -> Result<Json<BotejaoQueueWatcherResponse>, Custom<String>> {
    let yolo = match cached_yolo.try_write(){
        Ok(yolo) => yolo,
        Err(_err) => {
            return Err(Custom(Status::TooManyRequests,
                              "An image is already being processed, only one image can be processed at a time".to_string()));
        },
    };
    let img_url = "https://webservices.prefeitura.unicamp.br/cameras/cam_ra.jpg";
    let mut img_from_req = Vec::<u8>::new();
    match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
        .get(img_url)
        .send()
        {
            Ok(mut img) => {
                img.read_to_end(&mut img_from_req).unwrap();
            }
            Err(e) => {
                println!("{}", e.to_string());
                return Err(Custom(Status::InternalServerError,
                                  "Could not get image from camera using API (took more than 10s)".to_string()));
            }
        }
    let image_rust_ori = image::load_from_memory(img_from_req.as_slice())
        .unwrap()
        .to_rgb();
    let temp_img_filename = "rotated_and_cropped_img.png";
    let processed_img = rotate_and_crop(image_rust_ori);
    processed_img.save(temp_img_filename).unwrap();

    let nb_people = yolo.process_img().unwrap();
    let mut c = Cursor::new(Vec::new());
    let (width, height) = processed_img.dimensions();
    image::jpeg::JPEGEncoder::new(&mut c)
        .encode(&*processed_img, width, height, ColorType::RGB(8))
        .unwrap();
    c.set_position(0);
    let mut processed_jpg = Vec::new();
    File::open("processed_img.jpg")
        .unwrap()
        .read_to_end(&mut processed_jpg)
        .unwrap();
    let raw_bytes_to_send_as_b64 = base64::encode(&processed_jpg);
    use chrono::{DateTime, Local};
    let now: DateTime<Local> = Local::now();

    let date_str = now.format("%Y:%m:%d %H:%M:%S%:z").to_string();

    Ok(Json(BotejaoQueueWatcherResponse{
        number_of_people: nb_people,
        image_jpg_b64: raw_bytes_to_send_as_b64,
        last_update: date_str
    }))
}

#[derive(Serialize, Clone, Debug)]
struct BotejaoQueueWatcherResponse {
    number_of_people: u32,
    last_update: String,
    image_jpg_b64: String,
}

pub fn rotate_and_crop(image: image::RgbImage) -> image::RgbImage {
    let mut image_rust = imageproc::geometric_transformations::rotate_about_center(
        &image,
        0.3,
        Interpolation::Bicubic,
        image::Rgb([0u8, 0u8, 0u8]),
    );
    let sub_img = imageops::crop(&mut image_rust, 400, 0, 1080, 1080);
    sub_img.to_image()
}

pub fn main() -> failure::Fallible<()> {

    let yolo = Arc::new(RwLock::new(Yolo::new()
        .expect("Error creating an Yolo instance.")));

    rocket::ignite()
        .manage(yolo)
        .mount("/", routes![index])
        .launch();

    Ok(())
}
