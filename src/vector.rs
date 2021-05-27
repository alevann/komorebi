// Vector 2
type V2f32 = Vector<f32, 2>;
type V2f64 = Vector<f64, 2>;

// Vector 3
type V3f32 = Vector<f32, 3>;
type V3f64 = Vector<f64, 3>;

struct Vector<T, const S: usize> {
    data: [T; S],
}

macro_rules! impl_vector {
    ( $( $float:ty ),* ) => {
        $(
            impl<const S: usize> Vector<$float, S> {
                pub fn new() -> Vector<$float, S> {
                    Vector { data: [0.0; S] }
                }
                pub fn normalize(&mut self) {
                    let len = self.length();
                    for n in &mut self.data {
                        *n = *n / len;
                    }
                }
                fn length(&self) -> $float {
                    let mut length = 0.0;
                    for n in &self.data {
                        length += *n
                    }
                    length.sqrt()
                }
            }
        )*
    };
}

macro_rules! inst_vmacro {
    ( $type:ty, $name:expr ) => {
        macro_rules! v3f32 {
            ( $n1:expr, $n2:expr, $n3:expr ) => {{
                let mut temp = V3f32::new();
                temp.data[0] = $n1;
                temp.data[1] = $n2;
                temp.data[2] = $n3;
                temp
            }};
        }
    };
}

impl_vector!(f32, f64);
inst_vmacro!(V3f32, v3f32);

fn test() {
    let mut v = v3f32!(5.0, 5.0, 7.0);
    v.normalize();
}

struct Matrix<T, const C: usize, const R: usize> {
    data: [[T; C]; R],
}
