use tensorflow::{Graph, SavedModelBundle, Session, SessionOptions, SessionRunArgs, Tensor, Status};
use warp::{Rejection, reject::Reject};
use serde::{Deserialize, Serialize};
use std::result::Result;
use crate::BColors;

#[derive(Debug)]
struct CustomError(&'static str);

impl Reject for CustomError {}

fn model_load() -> Result<(Graph, Session), Status> {
    let mut graph = Graph::new();
    let session_opts = SessionOptions::new();
    let export_dir = "./model";
    let tags = ["serve"];
    let bundle = SavedModelBundle::load(&session_opts, &tags, &mut graph, export_dir)?;
    Ok((graph, bundle.session))
}

fn get_item_from_prediction(output: &[f32]) -> String {
    let items = vec!["Mechanical_Engineer", "Data_Analyst", "Veterinarian", "Sales_Representative", "Firefighter", "Biomedical_Researcher", "Software_Engineering", "Journalist", "Pilot", "Air_Traffic_Controller", "Psychologist", "Environmental_Scientist", "Real_Estate_Agent", "Interior_Designer", "Social_Worker", "Construction_Worker", "Nurse", "Astronomer", "Mechanic", "Architect", "Research_Scientist", "Accountant", "Pharmacist", "Civil_Engineer", "Chef", "Photographer", "Police_Officer", "Game_Developer", "Dentist", "Physical_Therapist", "Animator", "Financial_Advisor", "Fashion_Designer", "Surgeon", "Musician", "Tour_Guide", "Event_Planner", "Lawyer", "Plumber", "Electrician", "Teacher", "Marketing_Manager", "Graphic_Design", "Film_Director", "Athlete", "Farmer"];
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

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FormData {
    pub user_interests: Vec<String>,
    pub user_interest_encoder: String,
}

pub async fn get_post(id: u64, form: FormData) -> Result<impl warp::Reply, Rejection> {
    let colors = BColors::new();
    println!("{}[Rust AI] User interests: {}{:?}{}", colors.blue, colors.fail, form.user_interests, colors.endc);

    let (graph, session) = model_load().map_err(|_| warp::reject::custom(CustomError("Model loading failed")))?;

    let parsed_data: Vec<f32> = form.user_interest_encoder.split_whitespace().map(|s| s.parse().unwrap_or(0.0)).collect();
    let input_tensor = Tensor::new(&[1, parsed_data.len() as u64]).with_values(&parsed_data).map_err(|_| warp::reject::custom(CustomError("Tensor creation failed")))?;

    let mut args = SessionRunArgs::new();
    let input_op = graph.operation_by_name_required("serving_default_dense_input").map_err(|_| warp::reject::custom(CustomError("Input operation not found")))?;
    args.add_feed(&input_op, 0, &input_tensor);

    let output_op = graph.operation_by_name_required("StatefulPartitionedCall").map_err(|_| warp::reject::custom(CustomError("Output operation not found")))?;
    let output_token = args.request_fetch(&output_op, 0);

    session.run(&mut args).map_err(|_| warp::reject::custom(CustomError("Session run failed")))?;

    let output_tensor = args.fetch(output_token).map_err(|_| warp::reject::custom(CustomError("Fetching output tensor failed")))?;
    let output_data: Vec<f32> = output_tensor.iter().cloned().collect();
    let predicted_item = get_item_from_prediction(&output_data);

    println!("{}Prediction processed: {}{}", colors.blue, colors.endc, predicted_item);

    let post = Post {
        id,
        body: format!("Predicted item: {}", predicted_item),
    };
    Ok(warp::reply::json(&post))
}
