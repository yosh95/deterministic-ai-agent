use anyhow::Result;
use std::collections::HashMap;
use crate::model::{TokenClassifier, MasterMatcher};
use crate::encoder::EmbeddingEncoder;

pub struct ModelNER {
    classifier: TokenClassifier,
    matcher: MasterMatcher,
}

impl ModelNER {
    pub fn new(input_dim: usize, threshold: f32) -> Result<Self> {
        Ok(Self {
            classifier: TokenClassifier::new(input_dim, 3)?, // O, DEVICE, FAULT
            matcher: MasterMatcher::new(threshold),
        })
    }

    pub fn load_classifier(&mut self, path: &str) -> Result<()> {
        self.classifier.load_weights(path)?;
        Ok(())
    }

    pub fn register_device(&mut self, name: &str, encoder: &EmbeddingEncoder) -> Result<()> {
        let vec = encoder.encode(name)?;
        self.matcher.add_template(name, vec);
        Ok(())
    }

    pub fn extract(&self, text: &str, encoder: &EmbeddingEncoder) -> Result<HashMap<String, String>> {
        let hidden_states = encoder.get_hidden_states(text)?;
        let logits = self.classifier.forward(&hidden_states)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let labels_tensor = probs.argmax(1)?;
        let labels: Vec<u32> = labels_tensor.to_vec1()?;

        let tokenizer = encoder.get_tokenizer();
        let tokens = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        let token_ids = tokens.get_ids();
        let token_strings = tokens.get_tokens();

        let mut results = HashMap::new();
        let mut device_chunks = Vec::new();
        let mut fault_chunks = Vec::new();

        for (i, &label_id) in labels.iter().enumerate() {
            if i >= token_strings.len() { break; }
            let token_str = &token_strings[i];
            
            // Skip special tokens
            if (token_str.starts_with('[') && token_str.ends_with(']')) || token_ids[i] <= 103 {
                continue;
            }

            // High-priority Deterministic Match (RAG logic)
            let token_vec = hidden_states.get(i)?;
            if let Some((matched_id, _score)) = self.matcher.match_entity(&token_vec)? {
                results.insert("device_id".to_string(), matched_id);
            }

            // Buffer based on labels for reconstruction
            match label_id {
                1 => device_chunks.push(token_str.clone()),
                2 => fault_chunks.push(token_str.clone()),
                _ => {}
            }
        }

        // Reconstruct words from sub-tokens (WordPiece restoration)
        let reconstruct = |chunks: Vec<String>| -> String {
            let mut result = String::new();
            for chunk in chunks {
                if chunk.starts_with("##") {
                    result.push_str(&chunk[2..]);
                } else {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(&chunk);
                }
            }
            result
        };

        let device_name = reconstruct(device_chunks);
        let fault_name = reconstruct(fault_chunks);

        if !device_name.is_empty() {
            results.insert("device_candidate".to_string(), device_name);
        }
        if !fault_name.is_empty() {
            results.insert("fault_candidate".to_string(), fault_name);
        }

        Ok(results)
    }
}
