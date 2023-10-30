use warp::{Filter, Rejection, Reply};
use crate::BColors;

pub async fn handle_rejection(err: Rejection) -> Result<Box<dyn Reply>, Rejection> {
    let colors = BColors::new();
    let message = "Internal Server Error";
    eprintln!("{}[Rust AI]{} Unhandled Rejection:{} {:?}", colors.blue, colors.fail, colors.endc, err);
    let json = warp::reply::json(&serde_json::json!({"error": message}));
    Ok(Box::new(json))
}

fn get_post() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("predict" / u64)
        .and(warp::post())
        .and(warp::body::json())
        .and_then(super::route::get_post::get_post)
}

pub fn routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    get_post()
        .recover(handle_rejection)
}
