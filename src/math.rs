pub mod vector;
pub mod matrix;
pub mod angle;


pub fn bench_math() {
    println!("Benching Vector implementation");
    vector::__bench_vec();
    println!("Benching Matrix implementation");
    matrix::__bench_mat();
}