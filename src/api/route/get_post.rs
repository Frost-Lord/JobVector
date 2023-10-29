use warp::Rejection;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Post {
    pub id: u64,
    pub body: String,
}

// A function to handle GET requests at /posts/{id}
pub async fn get_post(id: u64) -> Result<impl warp::Reply, Rejection> {
    let post = Post {
        id,
        body: String::from("This is a post about Warp."),
    };
    Ok(warp::reply::json(&post))
}
