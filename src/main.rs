use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;
use std::fmt::{Display, Formatter, Result};

/// Преобразует цветовые компоненты пикселя к [`u8`] и печатает на экран.
///
/// [`u8`]: https://doc.rust-lang.org/std/primitive.u8.html
fn render_pixel<T>(pixel: T)
where
    T: Color,
{
    let ir = (255.99 * pixel.r()) as i32;
    let ig = (255.99 * pixel.g()) as i32;
    let ib = (255.99 * pixel.b()) as i32;

    println!("{} {} {}", ir, ig, ib);
}

/// Трехкомпонентный вектор с плавающей точкой.
#[derive(Copy, Clone, Debug)]
pub struct Vec3([f32; 3]);

impl Vec3 {
    /// Создает новый вектор.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { 0: [x, y, z] }
    }

    /// Возвращает длину вектора.
    pub fn length(&self) -> f32 {
        self.square_length().sqrt()
    }

    /// Возвращает квадрат длины вектора.
    pub fn square_length(&self) -> f32 {
        self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]
    }

    /// Приводит вектор к единичному виду.
    pub fn make_unit_vector(&mut self) {
        let k = 1.0 / self.length();
        *self *= k;
    }
}

/// Конструктор по умолчанию.
impl Default for Vec3 {
    fn default() -> Self {
        Vec3 { 0: [0.0, 0.0, 0.0] }
    }
}

/// Форматированный вывод.
impl Display for Vec3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{} {} {}", self.0[0], self.0[1], self.0[2])
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        self.0[0] == other.0[0] && self.0[1] == other.0[1] && self.0[2] == other.0[2]
    }
}

/// Унарный минус.
///
/// Применяет унарный минус к каждой компоненте вектора.
impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3 {
            0: [-self.0[0], -self.0[1], -self.0[2]],
        }
    }
}

/// Оператор чтения по индексу.
impl Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Оператор записи по индексу.
impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Оператор сложения векторов.
///
/// Векторы складываются по компонентам.
impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            0: [
                self.0[0] + rhs.0[0],
                self.0[1] + rhs.0[1],
                self.0[2] + rhs.0[2],
            ],
        }
    }
}

/// Оператор сложения вектора со скаляром.
///
/// К каждой компоненте вектора прибавляется число.
impl Add<f32> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: f32) -> Self::Output {
        Self {
            0: [
                self.0[0] + rhs,
                self.0[1] + rhs,
                self.0[2] + rhs,
            ],
        }
    }
}

/// Оператор сложения векторов с присваиванием.
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}

/// Оператор вычитания векторов.
///
/// Векторы вычитаются по компонентам.
impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            0: [
                self.0[0] - rhs.0[0],
                self.0[1] - rhs.0[1],
                self.0[2] - rhs.0[2],
            ],
        }
    }
}

/// Оператор вычитания вектора с присваиванием.
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
    }
}

/// Оператор умножения векторов.
///
/// Векторы умножаются по компонентам.
impl Mul for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            0: [
                self.0[0] * rhs.0[0],
                self.0[1] * rhs.0[1],
                self.0[2] * rhs.0[2],
            ],
        }
    }
}

/// Оператор умножения вектора на скаляр.
///
/// Каждая компонента вектора умножается на число.
impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self *= rhs;
        self
    }
}

/// Оператор умножения скаляра на вектор.
///
/// Каждая компонента вектора умножается на число.
impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

/// Оператор умножения на вектор с присваиванием.
impl MulAssign for Vec3 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0[0] *= rhs.0[0];
        self.0[1] *= rhs.0[1];
        self.0[2] *= rhs.0[2];
    }
}

/// Оператор деления векторов.
///
/// Компоненты первого вектора делятся на соответствующие компоненты второго вектора.
impl Div for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            0: [
                self.0[0] / rhs.0[0],
                self.0[1] / rhs.0[1],
                self.0[2] / rhs.0[2],
            ],
        }
    }
}

/// Оператор деления вектора на скаляр.
///
/// Каждая компонента вектора делится на число.
impl Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(mut self, rhs: f32) -> Self::Output {
        self /= rhs;
        self
    }
}

/// Оператор деления на вектор с присваиванием.
impl DivAssign for Vec3 {
    fn div_assign(&mut self, rhs: Self) {
        self.0[0] /= rhs.0[0];
        self.0[1] /= rhs.0[1];
        self.0[2] /= rhs.0[2];
    }
}

/// Оператор умножения на скаляр с присваиванием.
///
/// Каждая компонента вектора умножается на число.
impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
    }
}

/// Оператор деления на скаляр с присваиванием.
///
/// Каждая компонента вектора делится на число.
#[allow(clippy::suspicious_op_assign_impl)]
impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        let k = 1.0 / rhs;

        self.0[0] *= k;
        self.0[1] *= k;
        self.0[2] *= k;
    }
}

/// Скалярное произведение двух векторов. [Подробнее].
///
/// [Подробнее]: https://ru.wikipedia.org/wiki/%D0%A1%D0%BA%D0%B0%D0%BB%D1%8F%D1%80%D0%BD%D0%BE%D0%B5_%D0%BF%D1%80%D0%BE%D0%B8%D0%B7%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5
fn dot(a: Vec3, b: Vec3) -> f32 {
    a.0[0] * b.0[0] + a.0[1] * b.0[1] + a.0[2] * b.0[2]
}

/// Векторное произведение двух векторов. [Подробнее].
///
/// [Подробнее]: https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BD%D0%BE%D0%B5_%D0%BF%D1%80%D0%BE%D0%B8%D0%B7%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5
#[allow(dead_code)]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        0: [
            a.0[1] * b.0[2] - a.0[2] * b.0[1],
            -a.0[0] * b.0[2] - a.0[2] * b.0[0],
            a.0[0] * b.0[1] - a.0[1] * b.0[0],
        ],
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(a: [f32; 3]) -> Self {
        Self {0: a}
    }
}

/// Создает единичный вектор.
fn unit_vector(a: Vec3) -> Vec3 {
    a / a.length()
}

/// Типаж для описания вектора как точки в пространстве с координатами `x`, `y` и `z`.
trait Position {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn z(&self) -> f32;
}

/// Типаж для описания вектора как цвета с компонентами `r`, `g` и `b`.
trait Color {
    fn r(&self) -> f32;
    fn g(&self) -> f32;
    fn b(&self) -> f32;
}

impl Color for Vec3 {
    fn r(&self) -> f32 {
        self.0[0]
    }

    fn g(&self) -> f32 {
        self.0[1]
    }

    fn b(&self) -> f32 {
        self.0[2]
    }
}

impl Position for Vec3 {
    fn x(&self) -> f32 {
        self.0[0]
    }

    fn y(&self) -> f32 {
        self.0[1]
    }

    fn z(&self) -> f32 {
        self.0[2]
    }
}

/// Луч, направленный из точки `from` в точку `to`.
pub struct Ray {
    from: Vec3,
    to: Vec3,
}

impl Ray {
    /// Создает луч с началом в точке `from` и направленный в точку `to`.
    fn new(from: Vec3, to: Vec3) -> Self {
        Self {from, to}
    }

    /// Координаты точки, из которой исходит луч.
    fn origin(&self) -> Vec3 {
        self.from
    }

    /// Координаты точки, куда направлен луч.
    fn direction(&self) -> Vec3 {
        self.to
    }

    /// Координаты точки, лежащей на луче в отрезке `[from; to]`.
    ///
    /// Параметр `t` принимает значения в диапазоне `[0; 1]`.
    /// При `t = 0` точка совпадает с началом отрезка, при `t = 1` точка совпадает с концом отрезка.
    fn point_at_parameter(&self, t: f32) -> Vec3 {
        self.from + t * self.to
    }
}

/// Параметры попадания луча в объект.
struct HitRecord {
    t: f32,
    point: Vec3,
    normal: Vec3,
}

/// Типаж для реализации попадания луча в объект.
trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

/// Описывает положение и радиус сферы.
struct Sphere {
    center: Vec3,
    radius: f32,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = ray.origin() - self.center;
        let a = dot(ray.direction(), ray.direction());
        let b = 2.0 * dot(oc, ray.direction());
        let c = dot(oc, oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant > 0.0 {
            let t = (-b - discriminant.sqrt()) / 2.0 / a;
            if t > t_min && t < t_max {
                let point = ray.point_at_parameter(t);
                let normal = (point - self.center) / self.radius;

                return Some(HitRecord{ t, point, normal });
            }

            let t = (-b + discriminant.sqrt()) / 2.0 / a;
            if t > t_min && t < t_max {
                let point = ray.point_at_parameter(t);
                let normal = (point - self.center) / self.radius;

                return Some(HitRecord{ t, point, normal });
            }
        }

        None
    }
}

/// Массив объектов трехмерной сцены.
struct Scene(Vec<Box<dyn Hittable>>);

impl Hittable for Scene {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut rec: Option<HitRecord> = None;
        let mut closest_so_far = t_max;

        for object in &self.0 {
            if let Some(obj) = object.hit(ray, t_min, closest_so_far) {
                closest_so_far = obj.t;
                rec = Some(obj);
            }
        }
        rec
    }
}

/// Вычисляет цвет точки на экране.
fn pixel_color<T>(ray: &Ray, object: &T) -> Vec3
where
    T: Hittable
{
    let white: Vec3 = [1.0, 1.0, 1.0].into();
    let light_blue: Vec3 = [0.5, 0.7, 1.0].into();

    let record = object.hit(ray, 0.001, f32::MAX);
    match record {
        Some(hit) => {
            let target = hit.point + hit.normal + random_in_unit_sphere();
            let ray = Ray{ from: hit.point, to: target - hit.point };
            0.5 * pixel_color(&ray, object)
        },
        None => {
            let unit_direction = unit_vector(ray.direction());
            let t = 0.5 * (unit_direction.y() + 1.0);

            (1.0 - t) * white + t * light_blue
        }
    }
}

/// Описывает разрешение виртуального "экрана", на котором будет нарисовано финальное изображение.
struct Resolution {
    width: i32,
    height: i32,
}

/// Описывает параметры, необходимые для отрисовки финального изображения.
struct Camera {
    origin: Vec3,
    bottom_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    resolution: Resolution,
}

impl Default for Camera {
    // Для простоты кода в примере зададим все необходимые значения
    // в конструкторе по умолчанию.
    fn default() -> Self {
        Camera {
            origin: [0.0, 0.0, 0.0].into(),
            bottom_left: [-2.0, -1.0, -1.0].into(),
            horizontal: [4.0, 0.0, 0.0].into(),
            vertical: [0.0, 2.0, 0.0].into(),
            resolution: Resolution{ width: 0, height: 0 },
        }
    }
}

impl Camera {
    /// Создает камеру с экраном необходимого разрешения.
    fn with_resolution(res: Resolution) -> Self {
        let mut camera = Camera::default();

        camera.resolution = res;
        camera
    }

    /// Направляет луч из местоположения камеры в точку на виртуальном экране.
    fn direct_ray(&self, u: f32, v: f32) -> Ray {
        let to = self.bottom_left + u * self.horizontal + v * self.vertical;

        Ray::new(self.origin, to)
    }
}

use rand::distributions::{Distribution, Uniform};

/// Создает случайный вектор внутри единичной сферы методом исключения.
fn random_in_unit_sphere() -> Vec3 {
    let mut p;
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(0.0..1.0 as f32);

    loop {
        let x = dist.sample(& mut rng);
        let y = dist.sample(& mut rng);
        let z = dist.sample(& mut rng);

        p = 2.0 * Vec3::new(x, y, z) - Vec3::new(1.0, 1.0, 1.0);

        if p.square_length() < 1.0 {
            break;
        }
    }

    p
}

/// Усредняет цвет в окрестностях пикселя, делая `samples` выборок.
fn sampled_pixel_color<T>(camera: &Camera, object: &T, samples: i32, x: i32, y: i32) -> Vec3
where
    T: Hittable
{
    let mut color = Vec3::default();
    let mut rng = rand::thread_rng();
    let distribution = Uniform::from(0.0..1.0 as f32);
    let nx = camera.resolution.width;
    let ny = camera.resolution.height;

    for _ in 0..samples {
        let random = distribution.sample(& mut rng);
        let u = (x as f32 + random) / nx as f32;
        let v = ((ny - y) as f32  + random)/ ny as f32;

        let ray = camera.direct_ray(u, v);

        color += pixel_color(&ray, object);
    }

    color /= samples as f32;

    // Корректировка гаммы 1/2
    Vec3::new(color.r().sqrt(), color.g().sqrt(), color.b().sqrt())
}

fn main() {
    let width = 200;
    let height = 100;

    // Выводим заголовок формата PPM
    println!("P3");
    println!("{} {}", width, height);
    println!("255");

    let camera = Camera::with_resolution(Resolution{ width, height });
    let scene = Scene{
        0: vec![
            Box::new(Sphere{center: [0.0, 0.0, -1.0].into(), radius: 0.5}),
            Box::new(Sphere{center: [0.0, -100.5, -1.0].into(), radius: 100.0}),
        ]
    };

    let samples = 100;

    // Выводим остальные пиксели построчно начиная с верхнего левого угла изображения
    for y in 0..height {
        for x in 0..width {
            render_pixel(sampled_pixel_color(&camera, &scene, samples, x, y));
        }
    }
}
