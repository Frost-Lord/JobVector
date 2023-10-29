use std::error::Error;
use std::result::Result;
use tensorflow::{Graph, SavedModelBundle, Session, SessionOptions, SessionRunArgs, Tensor, Status};

mod api {
    pub mod routes;
    pub mod route {
        pub mod get_post;
    }
}

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

    let (graph, session) = model_load()?;
    
    let data: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let input_tensor = Tensor::new(&[1, 6]).with_values(&data)?;

    let mut args = SessionRunArgs::new();

    let input_op = graph.operation_by_name_required("serving_default_dense_input")?;
    args.add_feed(&input_op, 0, &input_tensor);

    let output_op = graph.operation_by_name_required("StatefulPartitionedCall")?;
    let output_token = args.request_fetch(&output_op, 0);
    
    session.run(&mut args)?;

    let output_tensor: Tensor<f32> = args.fetch(output_token)?;
    let output_data: Vec<f32> = output_tensor.iter().cloned().collect();
    println!("Model output: {:?}", output_data);

    let predicted_item = get_item_from_prediction(&output_data);
    println!("Predicted item: {}", predicted_item);

    Ok(())
}
