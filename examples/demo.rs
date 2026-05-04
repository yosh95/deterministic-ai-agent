use anyhow::Result;
use deterministic_ai_agent::encoder::EmbeddingEncoder;
use deterministic_ai_agent::ner::ModelNER;

fn main() -> Result<()> {
    // 1. Initialize Encoder
    println!("Initializing encoder...");
    let model_id = "sentence-transformers/all-MiniLM-L6-v2";
    let encoder = EmbeddingEncoder::new(model_id)?;
    let hidden_dim = 384; 

    // 2. Initialize Model-based NER (lowered threshold significantly for raw embedding demo)
    println!("Initializing Model-based NER...");
    let mut ner = ModelNER::new(hidden_dim, 0.5)?;

    // 3. Register Master Data
    println!("Registering master data...");
    ner.register_device("pump", &encoder)?;
    ner.register_device("valve", &encoder)?;

    // 4. Test deterministic extraction
    let input_text = "The water pump is failing.";
    println!("\nInput Text: \"{}\"", input_text);

    let results = ner.extract(input_text, &encoder)?;

    println!("\n--- Extraction Results ---");
    if results.is_empty() {
        println!("No entities found. (Try adjusting the confidence threshold)");
    } else {
        for (k, v) in results {
            println!("{}: {}", k, v);
        }
    }

    Ok(())
}
