use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use deterministic_ai_agent::AgentEngine;
use deterministic_ai_agent::encoder::EmbeddingEncoder;
use deterministic_ai_agent::model::{IntentClassifier, NERClassifier};
use deterministic_ai_agent::ner::NERExtractor;
use deterministic_ai_agent::train::Trainer;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TrainingItem {
    input: String,
    intent_id: u32,
    parameters: HashMap<String, serde_json::Value>,
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_dir = "models";
    fs::create_dir_all(model_dir)?;

    // 1. Initialize Encoder
    println!("--- Step 1: Initialize Encoder ---");
    let encoder = std::sync::Arc::new(EmbeddingEncoder::new("intfloat/multilingual-e5-small")?);
    let dim = 384;

    // 2. Load Data
    println!("\n--- Step 2: Load Training Data ---");
    let data_path = "data/sample_data.json";
    let content = fs::read_to_string(data_path)
        .with_context(|| format!("Failed to read {}", data_path))?;
    let items: Vec<TrainingItem> = serde_json::from_str(&content)?;
    println!("Loaded {} training items.", items.len());

    // 3. Train Intent Classifier
    println!("\n--- Step 3: Training Intent Classifier ---");
    let mut intent_embeddings = Vec::new();
    let mut intent_labels_vec = Vec::new();
    for item in &items {
        intent_embeddings.push(encoder.encode(&item.input)?);
        intent_labels_vec.push(item.intent_id);
    }
    let train_embeddings = Tensor::stack(&intent_embeddings, 0)?;
    let train_labels = Tensor::from_vec(intent_labels_vec.clone(), (items.len(),), &device)?;
    
    let num_intents = (intent_labels_vec.iter().max().unwrap_or(&0) + 1) as usize;
    let mut intent_classifier = IntentClassifier::new(dim, num_intents)?;
    let trainer = Trainer::new();

    trainer.train_intent(&mut intent_classifier, &train_embeddings, &train_labels, 200, 0.01)?;

    // Save Intent Weights & Centroids
    let intent_weights_path = format!("{}/intent_classifier.safetensors", model_dir);
    intent_classifier.varmap().save(&intent_weights_path)?;
    
    let centroids = trainer.calculate_centroids(&train_embeddings, &intent_labels_vec, num_intents)?;
    intent_classifier.set_centroids(centroids.clone());
    
    let centroid_path = format!("{}/centroids.safetensors", model_dir);
    let mut cmap = HashMap::new();
    cmap.insert("centroids".to_string(), centroids);
    candle_core::safetensors::save(&cmap, &centroid_path)?;
    println!("Intent Training completed and saved.");

    // 4. Train NER Classifier
    println!("\n--- Step 4: Training NER Classifier ---");
    let max_seq_len = 128;
    let mut ner_embeddings = Vec::new();
    let mut ner_labels = Vec::new();

    for item in &items {
        let tokens = encoder.get_tokens(&item.input)?;
        let labels = deterministic_ai_agent::ner::align_labels_with_tokens(&tokens, &item.parameters);
        let hidden_states = encoder.get_hidden_states(&item.input)?;

        let (seq_len, d) = hidden_states.dims2()?;
        if seq_len > max_seq_len { continue; }

        let pad_len = max_seq_len - seq_len;
        let padded_emb = if pad_len > 0 {
            let padding = Tensor::zeros((pad_len, d), hidden_states.dtype(), &device)?;
            Tensor::cat(&[&hidden_states, &padding], 0)?
        } else {
            hidden_states
        };

        let mut padded_labels = labels;
        padded_labels.resize(max_seq_len, 0);

        ner_embeddings.push(padded_emb);
        ner_labels.push(Tensor::new(padded_labels.as_slice(), &device)?);
    }

    let ner_emb_tensor = Tensor::stack(&ner_embeddings, 0)?;
    let ner_lab_tensor = Tensor::stack(&ner_labels, 0)?;
    let mut ner_classifier = NERClassifier::new(dim, 3)?; // O, DEVICE, FAULT

    trainer.train_ner(&mut ner_classifier, &ner_emb_tensor, &ner_lab_tensor, 200, 0.01)?;
    
    let ner_weights_path = format!("{}/ner_classifier.safetensors", model_dir);
    ner_classifier.varmap().save(&ner_weights_path)?;
    println!("NER Training completed and saved.");

    // 5. Initialize Engine and Run Inference
    println!("\n--- Step 5: Initializing Agent Engine ---");
    let mut ner_extractor = NERExtractor::new(dim, 0.85)?;
    ner_extractor.load_classifier(&ner_weights_path)?;

    // Register devices from config
    let devices_config_path = "config/devices.yaml";
    if Path::new(devices_config_path).exists() {
        let device_config: serde_yaml::Value = serde_yaml::from_str(&fs::read_to_string(devices_config_path)?)?;
        if let Some(devices) = device_config.get("devices").and_then(|d| d.as_sequence()) {
            for device_val in devices {
                if let Some(device_name) = device_val.as_str() {
                    ner_extractor.register_device(device_name, &encoder)?;
                }
            }
        }
    }

    let engine = AgentEngine::new(
        encoder, 
        intent_classifier, 
        ner_extractor, 
        Some("config/agent_settings.yaml")
    )?;

    // 6. Final Demo Inference
    let test_inputs = vec![
        "Warning: The Motor_B on line 5 show excessive vibration",
        "Conveyor_A のベルト異常振動を検出。診断を実行してください。",
        "What is the weather today?",
    ];

    println!("\n--- Final Demo ---");
    for input in test_inputs {
        let result = engine.run_step(input)?;
        println!("\nInput: '{}'", input);
        println!("  Intent ID: {}, Confidence: {:.4}", result.intent_id, result.confidence);
        println!("  Status: {}", result.status);
        if !result.parameters.is_empty() {
            println!("  Entities: {:?}", result.parameters);
        }
        if let Some(reason) = result.reason {
            println!("  Note: {}", reason);
        }
    }

    Ok(())
}
