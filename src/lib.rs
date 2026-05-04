pub mod encoder;
pub mod model;
pub mod ner;

use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use crate::encoder::EmbeddingEncoder;
use crate::model::IntentClassifier;
use crate::ner::NERExtractor;

#[derive(Debug, Deserialize)]
struct EngineConfig {
    engine: EngineSettings,
}

#[derive(Debug, Deserialize)]
struct EngineSettings {
    thresholds: Thresholds,
}

#[derive(Debug, Deserialize)]
struct Thresholds {
    confidence: f32,
    ood: f32,
}

pub struct AgentEngine {
    encoder: EmbeddingEncoder,
    classifier: IntentClassifier,
    ner: NERExtractor,
    confidence_threshold: f32,
    ood_threshold: f32,
}

impl AgentEngine {
    pub fn new(
        encoder: EmbeddingEncoder,
        classifier: IntentClassifier,
        ner: NERExtractor,
        config_path: Option<&str>,
    ) -> Self {
        let mut confidence_threshold = 0.70;
        let mut ood_threshold = 0.88;

        if let Some(path) = config_path {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(config) = serde_yaml::from_str::<EngineConfig>(&content) {
                    confidence_threshold = config.engine.thresholds.confidence;
                    ood_threshold = config.engine.thresholds.ood;
                }
            }
        }

        Self {
            encoder,
            classifier,
            ner,
            confidence_threshold,
            ood_threshold,
        }
    }

    pub fn run_step(&self, input: &str) -> Result<HashMap<String, String>> {
        let vector = self.encoder.encode(input)?;
        let (intent_id, confidence) = self.classifier.predict_with_confidence(&vector)?;
        let ood_score = self.classifier.get_ood_score(&vector)?;

        let mut output = HashMap::new();
        output.insert("input".into(), input.to_string());
        output.insert("intent_id".into(), intent_id.to_string());
        output.insert("confidence".into(), format!("{:.4}", confidence));
        output.insert("ood_score".into(), format!("{:.4}", ood_score));

        if ood_score < self.ood_threshold {
            output.insert("status".into(), "rejected_ood".into());
            output.insert("reason".into(), "Out of distribution".into());
            return Ok(output);
        }

        if confidence < self.confidence_threshold {
            output.insert("status".into(), "rejected_low_confidence".into());
            output.insert("reason".into(), "Low confidence score".into());
            return Ok(output);
        }

        let params = self.ner.extract(input);
        for (k, v) in params {
            output.insert(format!("param_{}", k), v);
        }
        output.insert("status".into(), "success".into());

        Ok(output)
    }
}
