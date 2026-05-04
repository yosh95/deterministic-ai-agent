use crate::encoder::EmbeddingEncoder;
use crate::model::{MasterMatcher, NERClassifier};
use anyhow::Result;
use std::collections::HashMap;

pub type NERExtractor = ModelNER;

pub struct ModelNER {
    classifier: NERClassifier,
    matcher: MasterMatcher,
}

impl ModelNER {
    pub fn new(input_dim: usize, threshold: f32) -> Result<Self> {
        Ok(Self {
            classifier: NERClassifier::new(input_dim, 3)?, // O, DEVICE, FAULT
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

    pub fn extract(
        &self,
        text: &str,
        encoder: &EmbeddingEncoder,
    ) -> Result<HashMap<String, String>> {
        let prefixed = format!("query: {}", text);
        let (hidden_states, _) = encoder.get_hidden_states_with_mask(&prefixed)?;
        let logits = self.classifier.forward(&hidden_states)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let labels_tensor = probs.argmax(1)?;
        let labels: Vec<u32> = labels_tensor.to_vec1()?;

        let tokenizer = encoder.get_tokenizer();
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let token_ids = tokens.get_ids();
        let token_strings = tokens.get_tokens();

        let mut results = HashMap::new();
        let mut device_chunks = Vec::new();
        let mut fault_chunks = Vec::new();

        for (i, &label_id) in labels.iter().enumerate() {
            if i >= token_strings.len() {
                break;
            }
            let token_str = &token_strings[i];

            // Skip special tokens
            if (token_str.starts_with('[') && token_str.ends_with(']')) || token_ids[i] <= 103 {
                continue;
            }

            // 1. High-priority Deterministic Match (Exact or Embedding similarity)
            let token_vec = hidden_states.get(i)?;
            if let Some((matched_id, _score)) = self.matcher.match_entity(token_str, &token_vec)? {
                results.insert("device_id".to_string(), matched_id);
            }

            // 2. Buffer based on classifier labels for reconstruction
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
                if let Some(stripped) = chunk.strip_prefix("##") {
                    result.push_str(stripped);
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

/// Utility to align labels for NER training with WordPiece awareness
pub fn align_labels_with_tokens(
    tokens: &[String],
    parameters: &HashMap<String, serde_json::Value>,
) -> Vec<u32> {
    let mut labels = vec![0u32; tokens.len()];

    // Create a normalized full text version from tokens for matching
    let mut current_pos = 0;
    let mut token_spans = Vec::new();
    for token in tokens {
        let clean = token.replace("##", "");
        let start = current_pos;
        let end = start + clean.len();
        token_spans.push((start, end, clean.to_lowercase()));
        current_pos = end;
    }

    for (key, val) in parameters {
        let val_str = match val {
            serde_json::Value::String(s) => s.to_lowercase(),
            serde_json::Value::Number(n) => n.to_string(),
            _ => continue,
        };

        let label_id = match key.as_str() {
            "device_id" | "item_name" => 1,
            "fault" => 2,
            _ => continue,
        };

        // Find tokens that build up the parameter value
        for (i, (_s, _e, token_text)) in token_spans.iter().enumerate() {
            if token_text.is_empty() {
                continue;
            }
            // Exact full token match or WordPiece part match
            if val_str.contains(token_text) && token_text.len() > 1 {
                labels[i] = label_id;
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_align_labels_with_tokens() {
        let tokens = vec![
            "the".to_string(),
            "motor".to_string(),
            "##_b".to_string(),
            "is".to_string(),
            "over".to_string(),
            "##heating".to_string(),
        ];

        let mut params = HashMap::new();
        params.insert("device_id".to_string(), json!("Motor_B"));
        params.insert("fault".to_string(), json!("overheating"));

        let labels = align_labels_with_tokens(&tokens, &params);

        // Expected:
        // the: 0
        // motor: 1 (DEVICE)
        // ##_b: 1 (DEVICE)
        // is: 0
        // over: 2 (FAULT)
        // ##heating: 2 (FAULT)
        assert_eq!(labels, vec![0, 1, 1, 0, 2, 2]);
    }

    #[test]
    fn test_align_labels_no_false_positive() {
        let tokens = vec!["remote".to_string()];
        let mut params = HashMap::new();
        params.insert("device_id".to_string(), json!("motor"));

        let labels = align_labels_with_tokens(&tokens, &params);

        // "remote" contains "mot", but should not match "motor"
        assert_eq!(labels, vec![0]);
    }
}
