use std::error::Error;
use std::result::Result;

mod api {
    pub mod routes;
    pub mod route {
        pub mod get_post;
    }
}

async fn run_server() -> Result<(), Box<dyn Error>> {
    let routes = api::routes::routes();
    println!("Server started at http://localhost:8000");
    warp::serve(routes).run(([127, 0, 0, 1], 8000)).await;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {

    tokio::runtime::Builder::new_multi_thread()
    .worker_threads(2)
    .enable_all()
    .build()
    .unwrap()
    .block_on(run_server())?;

    Ok(())
}
