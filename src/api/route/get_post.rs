use tensorflow::{Graph, SavedModelBundle, Session, SessionOptions, SessionRunArgs, Tensor, Status};
use warp::Rejection;
use serde::{Deserialize, Serialize};
use std::result::Result;
use crate::BColors;

fn model_load() -> Result<(Graph, Session), Status> {
    let mut graph = Graph::new();
    let session_opts = SessionOptions::new();
    let export_dir = "./saved_model";
    let tags = ["serve"];
    let bundle = SavedModelBundle::load(&session_opts, &tags, &mut graph, export_dir)?;
    Ok((graph, bundle.session))
}

fn get_item_from_prediction(output: &[f32]) -> String {
    let items = vec!["apple", "banana", "carrot", "donut", "egg", "fish"];
    let max_index = output.iter()
                          .enumerate()
                          .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                          .map(|(index, _)| index)
                          .unwrap_or(0);
    items[max_index].to_string()
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Post {
    pub id: u64,
    pub body: String,
}

pub async fn get_post(id: u64) -> Result<impl warp::Reply, Rejection> {
    let colors = BColors::new();
    let (graph, session) = model_load().unwrap();
    
    let data: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let input_tensor = Tensor::new(&[1, 6]).with_values(&data).unwrap();

    let mut args = SessionRunArgs::new();

    let input_op = graph.operation_by_name_required("serving_default_dense_input").unwrap();
    args.add_feed(&input_op, 0, &input_tensor);

    let output_op = graph.operation_by_name_required("StatefulPartitionedCall").unwrap();
    let output_token = args.request_fetch(&output_op, 0);
    
    session.run(&mut args).unwrap();

    let output_tensor: Tensor<f32> = args.fetch(output_token).unwrap();
    let output_data: Vec<f32> = output_tensor.iter().cloned().collect();

    let predicted_item = get_item_from_prediction(&output_data);
    println!("{}Prediction processed: {}{}", colors.blue, colors.endc, predicted_item);

    let post = Post {
        id,
        body: format!("Predicted item: {}", predicted_item),
    };
    Ok(warp::reply::json(&post))
}