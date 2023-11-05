use warp::{Filter, Rejection, Reply, http::StatusCode};
use crate::BColors;

pub async fn handle_rejection(err: Rejection) -> Result<Box<dyn Reply>, Rejection> {
    let colors = BColors::new();

    if let Some(e) = err.find::<warp::filters::body::BodyDeserializeError>() {
        let json = warp::reply::json(&serde_json::json!({
            "error": "Request body could not be deserialized",
            "message": format!("Could not deserialize the body: {}", e)
        }));
        return Ok(Box::new(warp::reply::with_status(json, StatusCode::BAD_REQUEST)));
    }

    let message = "Internal Server Error";
    eprintln!("{}[Rust AI]{} Unhandled Rejection:{} {:?}", colors.blue, colors.fail, colors.endc, err);
    let json = warp::reply::json(&serde_json::json!({"error": message}));
    Ok(Box::new(json))
}

fn get_post() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("predict" / u64)
        .and(warp::post())
        .and(
            warp::body::json()
                .or(warp::body::form())
                .unify()
        )
        .and_then(super::route::get_post::get_post)
}

mod routes_mod {
    use super::*;
    
    pub fn routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
        get_post()
            .recover(handle_rejection)
    }
}

pub use routes_mod::routes;
