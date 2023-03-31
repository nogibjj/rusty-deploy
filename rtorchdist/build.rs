fn main() {
    println!("cargo:rustc-link-search=native=/path/to/torch/library");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=dylib=torch");
}
