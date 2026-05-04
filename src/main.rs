use anyhow::Result;
use candle_core::{Device, Tensor};
use deterministic_ai_agent::encoder::EmbeddingEncoder;
use deterministic_ai_agent::model::IntentClassifier;
use deterministic_ai_agent::ner::NERExtractor;
use deterministic_ai_agent::AgentEngine;

fn main() -> Result<()> {
    // 1. Initialize Encoder (Downloads from HF)
    println!("Loading Encoder...");
    let encoder = EmbeddingEncoder::new("sentence-transformers/all-MiniLM-L6-v2")?;

    // 2. Initialize Classifier
    println!("Initializing Classifier...");
    let mut classifier = IntentClassifier::new(384, 5)?; // MiniLM l6 has 384 dim

    // 3. Prepare Dummy Training Data
    println!("Starting training simulation...");
    let device = Device::Cpu;
    
    // Label 0: Temperature problems
    // Label 1: Vibration problems
    let train_texts = vec![
        "The system is overheating",
        "High temperature detected",
        "Sensor reports overheat",
        "Too much vibration in the motor",
        "Excessive shaking on line 1",
    ];
    let train_labels_vec = vec![0u32, 0, 0, 1, 1];
    
    let embs: Vec<Tensor> = train_texts.iter()
        .map(|t| encoder.encode(t)).collect::<Result<Vec<_>>>()?;
    let train_embeddings = Tensor::stack(&embs, 0)?;
    let train_labels = Tensor::from_vec(train_labels_vec.clone(), (5,), &device)?;

    // 4. Training Loop
    for epoch in 1..=50 {
        let loss = classifier.train_one_epoch(&train_embeddings, &train_labels, 0.05)?;
        if epoch % 10 == 0 {
            println!("Epoch {:>2}/50, Loss: {:.6}", epoch, loss);
        }
    }

    // 5. Update Centroids for OOD detection
    classifier.update_centroids(&train_embeddings, &train_labels_vec)?;

    // 6. Initialize Engine with trained classifier
    println!("Loading NER...");
    let ner = NERExtractor::new("config/devices.yaml")?;
    let engine = AgentEngine::new(encoder, classifier, ner, Some("config/agent_settings.yaml"));

    // 7. Run inference
    let test_input = "Warning: The motor on line 5 show excessive vibration";
    println!("\nInference on: '{}'", test_input);
    let result = engine.run_step(test_input)?;

    println!("Result: {:#?}", result);

    Ok(())
}

