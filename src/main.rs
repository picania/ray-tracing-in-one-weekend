fn main() {
    let nx = 200;
    let ny = 100;

    // Выводим заголовок формата PPM
    println!("P3");
    println!("{} {}", nx, ny);
    println!("255");

    // Выводим остальные пиксели построчно начиная с верхнего левого угла изображения.
    for y in 0..ny {
        for x in 0..nx {
            let r = x as f32 / nx as f32;
            let g = (ny - y) as f32 / ny as f32;
            let b = 0.2f32;

            let ir = (255.99 * r) as i32;
            let ig = (255.99 * g) as i32;
            let ib = (255.99 * b) as i32;

            println!("{} {} {}", ir, ig, ib);
        }
    }
}
