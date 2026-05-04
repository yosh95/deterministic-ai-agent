use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use deterministic_ai_agent::encoder::EmbeddingEncoder;
use deterministic_ai_agent::train::Trainer;
use serde::Deserialize;
use std::fs;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct TrainingItem {
    input: String,
    parameters: HashMap<String, serde_json::Value>,
}

fn align_labels(tokens: &[String], parameters: &HashMap<String, serde_json::Value>) -> Vec<u32> {
    let mut labels = vec![0u32; tokens.len()]; 
    
    for (i, token) in tokens.iter().enumerate() {
        let clean_token = token.replace("##", "").to_lowercase();
        if clean_token.is_empty() || clean_token == "[cls]" || clean_token == "[sep]" {
            continue;
        }

        for (key, val) in parameters {
            let val_str = match val {
                serde_json::Value::String(s) => s.to_lowercase(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            
            if val_str.contains(&clean_token) || clean_token.contains(&val_str) {
                if key == "device_id" || key == "item_name" {
                    labels[i] = 1; // DEVICE
                } else if key == "fault" {
                    labels[i] = 2; // FAULT
                }
            }
        }
    }
    labels
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    
    // 1. Load data
    let data_path = "data/sample_data.json";
    let content = fs::read_to_string(data_path)
        .with_context(|| format!("Failed to read {}", data_path))?;
    let items: Vec<TrainingItem> = serde_json::from_str(&content)?;

    // 2. Initialize encoder
    let encoder = EmbeddingEncoder::new("sentence-transformers/all-MiniLM-L6-v2")?;

    // 3. Prepare dataset (Padding to max_seq_len for batching)
    let max_seq_len = 128; 
    let mut all_embeddings = Vec::new();
    let mut all_labels = Vec::new();

    println!("Vectorizing tokens for {} items...", items.len());
    for item in items {
        let hidden_states = encoder.get_hidden_states(&item.input)?; // [Seq, Dim]
        let tokens = encoder.get_tokens(&item.input)?;
        let labels = align_labels(&tokens, &item.parameters);

        let (seq_len, dim) = hidden_states.dims2()?;
        if seq_len > max_seq_len {
            println!("Skipping item (too long: {}): {}", seq_len, item.input);
            continue;
        }

        // Padding
        let pad_len = max_seq_len - seq_len;
        let padded_emb = if pad_len > 0 {
            let padding = Tensor::zeros((pad_len, dim), hidden_states.dtype(), &device)?;
            Tensor::cat(&[&hidden_states, &padding], 0)?
        } else {
            hidden_states
        };

        let mut padded_labels = labels;
        padded_labels.resize(max_seq_len, 0);

        all_embeddings.push(padded_emb);
        all_labels.push(Tensor::new(padded_labels.as_slice(), &device)?);
    }

    if all_embeddings.is_empty() {
        return Err(anyhow::anyhow!("No training data prepared. Check text lengths and max_seq_len."));
    }

    let embeddings_tensor = Tensor::stack(&all_embeddings, 0)?;
    let labels_tensor = Tensor::stack(&all_labels, 0)?;

    println!("NER Training Data Shape: {:?}", embeddings_tensor.shape());

    // 4. Run NER training
    let mut trainer = Trainer::new();
    let num_labels = 3; // O, DEVICE, FAULT
    let epochs = 150;
    let lr = 0.005;

    println!("Starting NER training...");
    trainer.train_ner(&embeddings_tensor, &labels_tensor, num_labels, epochs, lr)?;

    // 5. Save NER weights
    let save_path = "models/ner_classifier.safetensors";
    trainer.save_weights(save_path)?;
    println!("NER Training completed. Weights saved to: {}", save_path);

    Ok(())
}
