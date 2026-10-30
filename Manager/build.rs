// Alpha AI (AVA) — recursos do executável Windows (ícone) 
fn main() {
    println!("cargo:rerun-if-changed=assets/alpha.ico");

    // O build script roda no HOST; só gera recursos quando o TARGET é Windows.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let mut resource = winresource::WindowsResource::new();
    resource.set_icon("assets/alpha.ico");
    if let Err(error) = resource.compile() {
        // Não derruba o build do Linux se o windres não existir no host de
        // cross-compile; o ícone da JANELA continua funcionando via winit/tao.
        println!("cargo:warning=falha ao embutir o ícone do .exe: {error}");
    }
}
