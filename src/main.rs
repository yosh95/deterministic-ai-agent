use anyhow::Result;
use candle_core::{Device, Tensor};
use deterministic_ai_agent::AgentEngine;
use deterministic_ai_agent::encoder::EmbeddingEncoder;
use deterministic_ai_agent::model::IntentClassifier;
use deterministic_ai_agent::ner::NERExtractor;
use serde::Deserialize;

#[derive(Deserialize)]
struct TrainingSample {
    input: String,
    intent_id: u32,
}

fn main() -> Result<()> {
    // 1. Initialize Encoder (Multilingual model as per README)
    println!("Loading Multilingual Encoder...");
    let encoder = EmbeddingEncoder::new("intfloat/multilingual-e5-small")?;
    let dim = 384;

    // 2. Load Training Data from JSON
    println!("Loading training data from sample_data.json...");
    let data_content = std::fs::read_to_string("data/sample_data.json")?;
    let samples: Vec<TrainingSample> = serde_json::from_str(&data_content)?;

    let train_texts: Vec<String> = samples.iter().map(|s| s.input.clone()).collect();
    let train_labels_vec: Vec<u32> = samples.iter().map(|s| s.intent_id).collect();
    let num_intents = (train_labels_vec.iter().max().unwrap_or(&0) + 1) as usize;

    // 3. Initialize Classifier
    println!("Initializing Classifier for {} intents...", num_intents);
    let mut classifier = IntentClassifier::new(dim, num_intents)?;

    // 4. Encode Training Data
    println!("Encoding training samples (this may take a moment)...");
    let embs: Vec<Tensor> = train_texts
        .iter()
        .map(|t| encoder.encode(t))
        .collect::<Result<Vec<_>>>()?;
    let train_embeddings = Tensor::stack(&embs, 0)?;
    let train_labels = Tensor::from_vec(train_labels_vec.clone(), (samples.len(),), &Device::Cpu)?;

    // 5. Training Loop
    println!("Training...");
    for epoch in 1..=100 {
        let loss = classifier.train_one_epoch(&train_embeddings, &train_labels, 0.001)?;
        if epoch % 20 == 0 {
            println!("Epoch {:>3}/100, Loss: {:.6}", epoch, loss);
        }
    }

    // 6. Update Centroids for OOD detection
    classifier.update_centroids(&train_embeddings, &train_labels_vec)?;

    // 7. Initialize Engine
    println!("Loading NER and Engine...");
    let mut ner = NERExtractor::new(dim, 0.85)?;
    
    // Load devices from YAML and register them
    let device_config: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string("config/devices.yaml")?)?;
    if let Some(devices) = device_config.get("devices").and_then(|d| d.as_sequence()) {
        for device_val in devices {
            if let Some(device_name) = device_val.as_str() {
                ner.register_device(device_name, &encoder)?;
            }
        }
    }

    let engine = AgentEngine::new(encoder, classifier, ner, Some("config/agent_settings.yaml"))?;

    // 8. Run inference
    let test_inputs = vec![
        "Warning: The Motor_B on line 5 show excessive vibration",
        "Conveyor_A のベルト異常振動を検出。診断を実行してください。", // Japanese
        "What is the weather today?",                                  // Should be rejected as OOD
    ];

    for input in test_inputs {
        println!("\n--- Inference ---");
        println!("Input: '{}'", input);
        let result = engine.run_step(input)?;
        println!("Result Status: {}", result.status);
        if let Some(reason) = result.reason {
            println!("Reason: {}", reason);
        }
        println!(
            "Intent ID: {}, Confidence: {:.4}, OOD Score: {:.4}",
            result.intent_id, result.confidence, result.ood_score
        );
        if !result.parameters.is_empty() {
            println!("Params: {:?}", result.parameters);
        }
    }

    Ok(())
}
