use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use deterministic_ai_agent::encoder::EmbeddingEncoder;
use deterministic_ai_agent::train::Trainer;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
struct TrainingItem {
    input: String,
    intent_id: u32,
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    
    // 1. Load data
    let data_path = "data/sample_data.json";
    let content = fs::read_to_string(data_path)
        .with_context(|| format!("Failed to read {}", data_path))?;
    let items: Vec<TrainingItem> = serde_json::from_str(&content)?;

    println!("Loaded {} training items.", items.len());

    // 2. Initialize encoder
    let encoder = EmbeddingEncoder::new(
        "sentence-transformers/all-MiniLM-L6-v2",
    )?;

    // 3. Vectorize data
    let mut embeddings_list = Vec::new();
    let mut labels_list = Vec::new();

    for item in items {
        let vec = encoder.encode(&item.input)?;
        embeddings_list.push(vec);
        labels_list.push(item.intent_id);
    }

    let embeddings = Tensor::stack(&embeddings_list, 0)?;
    let labels = Tensor::new(labels_list.as_slice(), &device)?;

    println!("Dataset shape: {:?}", embeddings.shape());

    // 4. Run training
    let mut trainer = Trainer::new();
    let num_classes = 3; // intent_id: 0, 1, 2
    let epochs = 200;
    let lr = 0.01;

    println!("Starting training...");
    trainer.train_intent(&embeddings, &labels, num_classes, epochs, lr)?;

    // 5. Calculate and Save Centroids (for OOD detection)
    println!("Calculating class centroids...");
    let centroids = trainer.get_centroids(&embeddings, &labels_list, num_classes)?;
    
    // 6. Save specialized weights
    let model_dir = "models";
    fs::create_dir_all(model_dir)?;
    
    let model_path = format!("{}/intent_classifier.safetensors", model_dir);
    trainer.save_weights(&model_path)?;
    
    let centroid_path = format!("{}/centroids.safetensors", model_dir);
    let mut cmap = std::collections::HashMap::new();
    cmap.insert("centroids".to_string(), centroids);
    candle_core::safetensors::save(&cmap, &centroid_path)?;

    println!("Training completed.");
    println!("Weights saved to: {}", model_path);
    println!("Centroids saved to: {}", centroid_path);

    Ok(())
}
