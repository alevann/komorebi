pub enum Angle {
    Deg(f32),
    Rad(f32)
}

impl Angle {
    pub fn as_rad(self) -> f32 {
        match self {
            Angle::Deg(a) => a.to_radians(),
            Angle::Rad(a) => a
        }
    }
}
