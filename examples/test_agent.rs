use anyhow::Result;
use deterministic_ai_agent::encoder::EmbeddingEncoder;
use deterministic_ai_agent::model::IntentClassifier;
use deterministic_ai_agent::ner::ModelNER;
use deterministic_ai_agent::AgentEngine;

fn main() -> Result<()> {
    // 1. Initialize Encoder
    println!("Initializing encoder...");
    let model_id = "sentence-transformers/all-MiniLM-L6-v2";
    let encoder = EmbeddingEncoder::new(model_id)?;
    let hidden_dim = 384; 

    // 2. Initialize Classifiers
    println!("Initializing classifiers...");
    let mut intent_classifier = IntentClassifier::new(hidden_dim, 3)?;
    
    // Load trained weights if available
    let weights_path = "models/intent_classifier.safetensors";
    let centroid_path = "models/centroids.safetensors";
    if std::path::Path::new(weights_path).exists() {
        println!("Loading trained weights from {}...", weights_path);
        intent_classifier.load_weights(weights_path)?;
        
        if std::path::Path::new(centroid_path).exists() {
            println!("Loading centroids for OOD detection...");
            let device = candle_core::Device::Cpu;
            let tensors = candle_core::safetensors::load(centroid_path, &device)?;
            if let Some(centroids) = tensors.get("centroids") {
                intent_classifier.set_centroids(centroids.clone());
            }
        }
    } else {
        println!("Warning: No trained weights found. Using random initialization.");
    }

    // 3. Initialize NER
    let mut ner = ModelNER::new(hidden_dim, 0.6)?;
    
    // Load trained NER weights if available
    let ner_weights_path = "models/ner_classifier.safetensors";
    if std::path::Path::new(ner_weights_path).exists() {
        println!("Loading trained NER weights from {}...", ner_weights_path);
        ner.load_classifier(ner_weights_path)?;
    }

    ner.register_device("Motor_B", &encoder)?;
    ner.register_device("Conveyor_A", &encoder)?;
    ner.register_device("Pump_01", &encoder)?;

    // 4. Setup Agent Engine
    let engine = AgentEngine::new(encoder, intent_classifier, ner, None)?;

    // 5. Run Inference
    let test_inputs = vec![
        "Motor_B shows signs of overheating. Diagnostic check required immediately.",
        "How many Sensor_C units are available in the warehouse?",
        "Routine status log: all systems nominal on line 1.",
        "Unknown random input about something else.",
    ];

    println!("\n--- Testing Agent Pipeline ---");
    for input in test_inputs {
        let result = engine.run_step(input)?;
        println!("\nInput: {}", result.input);
        println!("Intent ID: {} (Confidence: {:.2})", result.intent_id, result.confidence);
        println!("Status: {}", result.status);
        if !result.parameters.is_empty() {
            println!("Params: {:?}", result.parameters);
        }
        if let Some(r) = result.reason {
            println!("Reason: {}", r);
        }
    }

    Ok(())
}
