fn main() {
    println!("cargo:rustc-link-search=native={}", std::env::var("LIBTORCH").unwrap());
    println!("cargo:rustc-link-search=native={}/lib", std::env::var("LIBTORCH").unwrap());
}
