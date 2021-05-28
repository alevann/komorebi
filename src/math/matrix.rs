use std::ops::{ Mul, Index, IndexMut };
use std::cmp::PartialEq;
use float_cmp::{ ApproxEq, F32Margin };
use super::vector::Vec3;
use super::angle::Angle;

#[derive(Debug, Copy, Clone)]
pub struct Matrix<T, const C: usize, const R: usize> {
    pub data: [[T; C]; R]
}
pub type Mat4 = Matrix<f32, 4, 4>;

// TODO: unsure if scale_matrix and trans_matrix 
//  should be here or in transform

impl Mat4 {
    pub fn new() -> Self { 
        Self { data: [[0.0; 4]; 4] }
    }

    pub fn model_matrix() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
    }

    pub fn from_scale_and_trans(s: &Vec3, t: &Vec3) -> Self {
        Self {
            data: [
                [s[0], 0.0 , 0.0 , 0.0],
                [0.0 , s[1], 0.0 , 0.0],
                [0.0 , 0.0 , s[2], 0.0],
                [t[0], t[1], t[2], 1.0]
            ]
        }
    }

    pub fn scale_matrix(matrix: &mut Self, scale: Vec3) {
        matrix[0][0] = scale[0];
        matrix[1][1] = scale[1];
        matrix[2][2] = scale[2];
    }

    pub fn trans_matrix(matrix: &mut Self, trans: Vec3) {
        matrix[3][0] = trans[0];
        matrix[3][1] = trans[1];
        matrix[3][2] = trans[2];
    }

    /// # Rotate Slow
    /// 
    /// Returns a new matrix on which a rotation has been applied.
    /// Rotation indicates the direction in which to apply the rotation
    /// while angle determines the amount of rotation to apply.
    pub fn rsw(self, angle: Angle, rotation: &Vec3) -> Self {
        let a = angle.as_rad();
        let c = a.cos();
        let s = a.sin();
        let axis = rotation.nrm();

        let mut m = Self::new();

        m[0][0] = c + (1.0 - c) * axis[0] * axis[0];
        m[0][1] = (1.0 - c)     * axis[0] * axis[1] + s * axis[2];
        m[0][2] = (1.0 - c)     * axis[0] * axis[2] - s * axis[1];
        m[0][3] = 0.0;

        m[1][0] = (1.0 - c)     * axis[1] * axis[0] - s * axis[2];
        m[1][1] = c + (1.0 - c) * axis[1] * axis[1];
        m[1][2] = (1.0 - c)     * axis[1] * axis[2] + s * axis[0];
        m[1][3] = 0.0;

        m[2][0] = (1.0 - c)     * axis[2] * axis[0] + s * axis[1];
        m[2][1] = (1.0 - c)     * axis[2] * axis[1] - s * axis[0];
        m[2][2] = c + (1.0 - c) * axis[2] * axis[2];
        m[2][3] = 0.0;

        m[3][0] = 0.0;
        m[3][1] = 0.0;
        m[3][2] = 0.0;
        m[3][3] = 1.0;

        m * self
    }
}
        
impl Mul for Mat4 {
    type Output = Self;

    fn mul(self, o: Self) -> Self {
        let mut out = Self::new();

        out[0][0] = self[0][0]*o[0][0] + self[0][1]*o[1][0] + self[0][2]*o[2][0] + self[0][3]*o[3][0];
        out[0][1] = self[0][0]*o[0][1] + self[0][1]*o[1][1] + self[0][2]*o[2][1] + self[0][3]*o[3][1];
        out[0][2] = self[0][0]*o[0][2] + self[0][1]*o[1][2] + self[0][2]*o[2][2] + self[0][3]*o[3][2];
        out[0][3] = self[0][0]*o[0][3] + self[0][1]*o[1][3] + self[0][2]*o[2][3] + self[0][3]*o[3][3];

        out[1][0] = self[1][0]*o[0][0] + self[1][1]*o[1][0] + self[1][2]*o[2][0] + self[1][3]*o[3][0];
        out[1][1] = self[1][0]*o[0][1] + self[1][1]*o[1][1] + self[1][2]*o[2][1] + self[1][3]*o[3][1];
        out[1][2] = self[1][0]*o[0][2] + self[1][1]*o[1][2] + self[1][2]*o[2][2] + self[1][3]*o[3][2];
        out[1][3] = self[1][0]*o[0][3] + self[1][1]*o[1][3] + self[1][2]*o[2][3] + self[1][3]*o[3][3];

        out[2][0] = self[2][0]*o[0][0] + self[2][1]*o[1][0] + self[2][2]*o[2][0] + self[2][3]*o[3][0];
        out[2][1] = self[2][0]*o[0][1] + self[2][1]*o[1][1] + self[2][2]*o[2][1] + self[2][3]*o[3][1];
        out[2][2] = self[2][0]*o[0][2] + self[2][1]*o[1][2] + self[2][2]*o[2][2] + self[2][3]*o[3][2];
        out[2][3] = self[2][0]*o[0][3] + self[2][1]*o[1][3] + self[2][2]*o[2][3] + self[2][3]*o[3][3];

        out[3][0] = self[3][0]*o[0][0] + self[3][1]*o[1][0] + self[3][2]*o[2][0] + self[3][3]*o[3][0];
        out[3][1] = self[3][0]*o[0][1] + self[3][1]*o[1][1] + self[3][2]*o[2][1] + self[3][3]*o[3][1];
        out[3][2] = self[3][0]*o[0][2] + self[3][1]*o[1][2] + self[3][2]*o[2][2] + self[3][3]*o[3][2];
        out[3][3] = self[3][0]*o[0][3] + self[3][1]*o[1][3] + self[3][2]*o[2][3] + self[3][3]*o[3][3];
        
        out
    }
}

impl Index<usize> for Mat4 {
    type Output = [f32; 4];

    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i]
    }
}

impl PartialEq for Mat4 {
    fn eq(&self, other: &Self) -> bool {
        let margin = F32Margin { ulps: 5, epsilon: 0.0 };
        self[0][0].approx_eq(other[0][0], margin)
        && self[0][1].approx_eq(other[0][1], margin)
        && self[0][2].approx_eq(other[0][2], margin)
        && self[0][3].approx_eq(other[0][3], margin)
        && self[1][0].approx_eq(other[1][0], margin)
        && self[1][1].approx_eq(other[1][1], margin)
        && self[1][2].approx_eq(other[1][2], margin)
        && self[1][3].approx_eq(other[1][3], margin)
        && self[2][0].approx_eq(other[2][0], margin)
        && self[2][1].approx_eq(other[2][1], margin)
        && self[2][2].approx_eq(other[2][2], margin)
        && self[2][3].approx_eq(other[2][3], margin)
        && self[3][0].approx_eq(other[3][0], margin)
        && self[3][1].approx_eq(other[3][1], margin)
        && self[3][2].approx_eq(other[3][2], margin)
        && self[3][3].approx_eq(other[3][3], margin)
    }
}


#[test]
fn test_matrix_mul() {
    let m1 = Mat4 {
        data: [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    };
    let m2 = Mat4::model_matrix();
    assert_eq!(m1*m2, Mat4 {
        data: [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    });

    let m1 = Mat4 {
        data: [
            [15.0 , 32.002, 11.2 , -15.34001],
            [84.7 , -5.0  , 4.009, 6.3      ],
            [66.89, 545.3 , 11.25, 98.0032  ],
            [-48.5, 43.0  , 453.0, -84.00545]
        ]
    };
    let m2 = Mat4 {
        data: [
            [781.1, 11.002, 4.924, 911.2254 ],
            [554.3, -864.3, 11.5 , 2.665    ],
            [84.52, 94.255, 47.23, 435.1135 ],
            [112.5, 554.12, 4.664, 666.666  ]
        ]
    };
    assert_eq!(m1*m2, Mat4 {
        data: [
            [28676.081475, -34938.8489412, 899.31319336 , 8400.27442334 ],
            [64435.26068 ,  9122.193695  , 578.29107    , 83111.8322015 ],
            [366483.779  , -415200.964286, 7588.7407848 , 132635.5197122],
            [14788.496875, -41550.081954 , 21259.0745812, 97023.0012703 ],
        ]
    });
}


pub fn __bench_mat() {
    const Q: usize = 1_000_000;
    let now = std::time::Instant::now();
    for i in 0..Q {
        Mat4::model_matrix() * Mat4::new();
    }
    let els = now.elapsed();
    println!("Multiplied {} matrices together in {:.4?}", Q, els);
}