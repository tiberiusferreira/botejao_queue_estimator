#![feature(proc_macro_hygiene, decl_macro)]

extern crate darknet;
use darknet::*;
use image::{imageops, ColorType, Pixel, Rgb, RgbImage};
use imageproc::geometric_transformations::Interpolation;
use imageproc::rect::Rect;
use imageproc::rect::Region;
use std::fs::File;
use std::io::{Cursor, Write};
use std::process::exit;
use std::time::Instant;

#[macro_use] extern crate rocket;
fn main() {
    run();
}

pub fn rotate_and_crop(image: image::RgbImage) -> image::RgbImage {
    let mut image_rust = imageproc::geometric_transformations::rotate_about_center(
        &image,
        0.3,
        Interpolation::Nearest,
        image::Rgb([0u8, 0u8, 0u8]),
    );
    let sub_img = imageops::crop(&mut image_rust, 400, 0, 1080, 1080);
    sub_img.to_image()
}

pub struct YoloNetwork {
    network: Network,
    meta: Meta,
}

impl YoloNetwork {
    pub fn new() -> Self {
        let network = Network::new("./yolov3_reduced.cfg", "yolov3.weights").unwrap();
        let meta = Meta::new("cfg/coco.data").unwrap();
        YoloNetwork { network, meta }
    }

    pub fn detect(&self, image: &Image) -> Vec<Detection> {
        simple_detect(&self.network, &self.meta, image)
    }
}

pub fn filter_only_people_in_queue_area(dets: Vec<Detection>) -> Vec<Detection> {
    return dets
        .into_iter()
        .filter(|det| det.class == 0 && det.y < 900.0)
        .collect();
}

pub fn draw_dets_boxes(dets: Vec<Detection>, image: &mut RgbImage) {
    for det in dets {
        let rect = Rect::at((det.x - det.w / 2.0) as i32, (det.y - det.h / 2.0) as i32)
            .of_size(det.w as u32, det.h as u32);
        imageproc::drawing::draw_filled_rect_mut(image, rect, Rgb([255u8, 0u8, 0u8]));
    }
}

fn run() {
    let yolo = YoloNetwork::new();


    std::thread::spawn(||{
        loop {
            let img_path = "test_full.jpg";

            let image_rust_ori = image::open(img_path)
                .expect("No image found at provided path")
                .to_rgb();
            let temp_img_filename = "rotated_and_cropped_img.png";
            let mut processed_img = rotate_and_crop(image_rust_ori);
            processed_img.save(temp_img_filename).unwrap();

            let image = Image::load(temp_img_filename).unwrap();
            let dets = yolo.detect(&image);
            let people_dets = filter_only_people_in_queue_area(dets);
            draw_dets_boxes(people_dets, &mut processed_img);
            let mut c = Cursor::new(Vec::new());
            let (width, height) = processed_img.dimensions();
            image::jpeg::JPEGEncoder::new(&mut c)
                .encode(&*processed_img, width, height, ColorType::RGB(8))
                .unwrap();
            c.set_position(0);
            let raw_bytes_to_send = c.into_inner().as_slice();
            processed_img.save("out.png").unwrap();
        }
    });
    // would receive from network here


    rocket::ignite().mount("/", routes![index]).launch();

}


#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

//#[cfg(feature = "nnpack")]
//fn run() -> Result<(), Error> {
//    std::env::set_current_dir("darknet-sys").unwrap();
//
//    let mut network = Network::new("yolov3_reduced.cfg", "yolov3.weights")?;
//    network.create_threadpool(4);
//    let start = std::time::Instant::now();
//    let meta = Meta::new("cfg/coco.data")?;
//    let mut image = Image::load_threaded("data/dog.jpg", network.channel(), &network.threadpool())?;
//    let dets = simple_detect(&network, &meta, &image);
//    println!("Took: {} ms", start.elapsed().as_millis());
//    for d in &dets {
//        image.draw_box(d, 1, 1.0, 0.0, 0.0);
//    }
//    println!("{:?}", dets);
//
//    let data = image.encode_jpg();
//    {
//        let mut buffer = ::std::fs::File::create("prediction.jpg")?;
//        buffer.write_all(&data)?;
//    }
//    Ok(())
//}
