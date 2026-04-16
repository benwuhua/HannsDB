use std::path::PathBuf;

use hannsdb_daemon::routes::build_router;
use tokio::net::TcpListener;

fn parse_args() -> (PathBuf, u16) {
    let args: Vec<String> = std::env::args().collect();
    let mut data_dir = PathBuf::from("./hannsdb_data");
    let mut port: u16 = 19530;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" | "-d" => {
                i += 1;
                data_dir = PathBuf::from(args[i].as_str());
            }
            "--port" | "-p" => {
                i += 1;
                port = args[i].parse().expect("invalid port number");
            }
            "--help" | "-h" => {
                eprintln!("Usage: hannsdb-daemon [--data-dir <path>] [--port <port>]");
                eprintln!("  --data-dir, -d  Database root directory (default: ./hannsdb_data)");
                eprintln!("  --port, -p      HTTP listen port (default: 19530)");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (data_dir, port)
}

#[tokio::main]
async fn main() {
    let (data_dir, port) = parse_args();

    println!("hannsdb-daemon starting");
    println!("  data-dir: {}", data_dir.display());
    println!("  port:     {}", port);

    let app = build_router(&data_dir).expect("failed to open database");

    let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("failed to bind port");
    println!("  listening on 0.0.0.0:{port}");

    axum::serve(listener, app).await.expect("server error");
}
