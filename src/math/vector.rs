use std::ops::{Index};
use std::cmp::{PartialEq};
use std::fmt::{Display,Formatter,Result};
use rand::Rng;
use float_cmp::{ApproxEq,F32Margin};

#[derive(Debug)]
pub struct Vector<T, const S: usize> {
    data: [T; S],
}
pub type Vec3 = Vector<f32, 3>;

impl Vec3 {
    pub fn new() -> Self {
        Vector { data: [0.0; 3] }
    }

    pub fn up() -> Self {
        Vector { data: [ 0.0, 1.0, 0.0 ] }
    }

    pub fn nrm(&self) -> Self {
        let len = self.mag();
        Vector {
            data: [
                self[0] / len,
                self[1] / len,
                self[2] / len
            ]
        }
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self[0]*other[0] + self[1]*other[1] + self[2]*other[2]
    }

    #[inline(always)]
    fn mag(&self) -> f32 {
        let mut mgn = 0.0;
        mgn += self[0] * self[0];
        mgn += self[1] * self[1];
        mgn += self[2] * self[2];
        mgn.sqrt()
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;
    
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        let margin = F32Margin { ulps: 5, epsilon: 0.0 };
        self[0].approx_eq(other[0], margin)
        && self[1].approx_eq(other[1], margin)
        && self[2].approx_eq(other[2], margin)
    }
}

impl Display for Vec3 {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Vector3 {{ {:.6}, {:.6}, {:.6} }}", 
                self[0], self[1], self[2])
    }
}

#[macro_export]
macro_rules! vec3 {
    ( $n1:expr, $n2:expr, $n3:expr ) => {
        {
            Vector::<f32, 3> {
                data: [
                    $n1,
                    $n2,
                    $n3
                ]
            }
        }
    };
}


#[test]
fn test_vector_mag() {
    let v = vec3!(1.0, 1.0, 1.0);
    assert_eq!(v.mag(), 1.7320508);

    let v = vec3!(2.0, 2.0, 2.0);
    assert_eq!(v.mag(), 3.4641016);

    let v = vec3!(6.321, 147.54001, -3.052);
    assert_eq!(v.mag(), 147.70688);
}

#[test]
fn test_vector_nrm() {
    let v = vec3!(1.0, 1.0, 1.0).nrm();
    assert_eq!(v, vec3!(0.57735026, 0.57735026, 0.57735026));

    let v = vec3!(3.0, 8.0, 5.0).nrm();
    assert_eq!(v, vec3!(0.30304576, 0.80812203, 0.50507627));

    let v = vec3!(643.22115087, 551.00000012, 73.15634897).nrm();
    assert_eq!(v, vec3!(0.75663322, 0.64815173, 0.08605519));
}

#[test]
fn test_vector_dot() {
    let v = vec3!(1.0, 1.0, 1.0).dot(&vec3!(1.0, 1.0, 1.0));
    assert_eq!(v, 3.0);

    let v = vec3!(5.0, 64.0, 11.0).dot(&vec3!(-6.0, 73.0, 542.0));
    assert_eq!(v, 10604.0);
}


pub fn __bench_vec() {
    let mut rng = rand::thread_rng();

    let mut data = Vec::new();
    const Q: usize = 100_000_0;
    for _ in 0..Q  {
        data.push(vec3!(rng.gen(), rng.gen(), rng.gen()));
    }

    println!("Calling each function {} times", Q);

    let now = std::time::Instant::now();
    for i in 0..Q {
        data[i].mag();
    }
    let el = now.elapsed();
    println!("Vec.mag: {:.4?}", el);

    let now = std::time::Instant::now();
    for i in 0..Q {
        data[i].nrm();
    }
    let el = now.elapsed();
    println!("Vec.nrm: {:.4?}", el);

    let now = std::time::Instant::now();
    for i in 0..Q-1 {
        data[i].dot(&data[i+1]);
    }
    let el = now.elapsed();
    println!("Vec.dot: {:.4?}", el);
}