use warp::Filter;

pub fn routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    get_post()
}

fn get_post() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("predict" / u64)
        .and(warp::get())
        .and_then(super::route::get_post::get_post)
}
