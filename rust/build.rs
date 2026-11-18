//! With the "signalsmith" feature (C++), link the C++ runtime statically on MinGW so
//! lls-dsp.exe needs no libstdc++/libgcc/winpthread DLLs (.cargo/config.toml stops cc
//! from linking libstdc++ itself). rustc puts these after its own libraries.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let gnu = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu");
    if std::env::var_os("CARGO_FEATURE_SIGNALSMITH").is_none() {
        return;
    }
    let libs = if gnu && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        "-Wl,-Bstatic -lstdc++ -lgcc -lgcc_eh -lwinpthread -Wl,-Bdynamic -lmingwex -lmsvcrt -lkernel32"
    } else if gnu {
        "-lstdc++"
    } else {
        return;
    };
    for a in libs.split(' ') {
        println!("cargo:rustc-link-arg-bins={a}");
    }
}
