use super::matrix::Mat4;
use super::vector::{Vec3,Vector, self};
use super::angle::Angle;
use crate::vec3;

pub struct Transform {
    pub matrix: Mat4,
    // TODO: rotation should be a quaternion
    rotation: Vec3,
    position: Vec3,
    scale: Vec3
}

impl Transform {
    pub fn new() -> Self {
        // FIXME: position and rotation are Vector3 zero
        //  only because model_matrix makes them be zero
        Self {
            matrix: Mat4::model_matrix(),
            rotation: Vec3::new(),
            position: Vec3::new(),
            scale: vec3!(1.0, 1.0, 1.0)
        }
    }


    pub fn set_scale(&mut self, s: Vec3) {
        Mat4::scale_matrix(&mut self.matrix, s);
        self.scale = s;
    }


    pub fn set_position(&mut self, p: Vec3) {
        Mat4::trans_matrix(&mut self.matrix, p);
        self.position = p;
    }


    pub fn set_rotation(&mut self, r: Vec3) {
        let mat = Mat4::from_scale_and_trans(&self.scale, &self.position);
        let mat = mat.rsw(Angle::Deg(r[0]), &vector::RG);
        let mat = mat.rsw(Angle::Deg(r[1]), &vector::UP);
        let mat = mat.rsw(Angle::Deg(r[2]), &vector::FW);
        self.matrix = mat;
        self.rotation = r;
    }
}
