pub mod encoder;
pub mod model;
pub mod ner;
pub mod train;

use crate::encoder::EmbeddingEncoder;
use crate::model::IntentClassifier;
use crate::ner::ModelNER;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

#[derive(Debug, Serialize)]
pub struct AgentResult {
    pub input: String,
    pub intent_id: u32,
    pub confidence: f32,
    pub ood_score: f32,
    pub status: String,
    pub reason: Option<String>,
    pub parameters: HashMap<String, String>,
}

pub struct AgentEngine {
    encoder: EmbeddingEncoder,
    classifier: IntentClassifier,
    ner: ModelNER,
    confidence_threshold: f32,
    ood_threshold: f32,
}

impl AgentEngine {
    pub fn new(
        encoder: EmbeddingEncoder,
        classifier: IntentClassifier,
        ner: ModelNER,
        config_path: Option<&str>,
    ) -> Result<Self> {
        let mut confidence_threshold = 0.70;
        let mut ood_threshold = 0.88;

        if let Some(path) = config_path {
            let content = std::fs::read_to_string(path)
                .map_err(|e| anyhow!("Failed to read config file at {}: {}", path, e))?;
            let config: EngineConfig = serde_yaml::from_str(&content)
                .map_err(|e| anyhow!("Failed to parse config YAML: {}", e))?;

            confidence_threshold = config.engine.thresholds.confidence;
            ood_threshold = config.engine.thresholds.ood;
        }

        Ok(Self {
            encoder,
            classifier,
            ner,
            confidence_threshold,
            ood_threshold,
        })
    }

    pub fn run_step(&self, input: &str) -> Result<AgentResult> {
        let vector = self.encoder.encode(input)?;
        let (intent_id, confidence) = self.classifier.predict_with_confidence(&vector)?;
        let ood_score = self.classifier.get_ood_score(&vector)?;

        let mut status = "success".to_string();
        let mut reason = None;
        let mut params = HashMap::new();

        if ood_score < self.ood_threshold {
            status = "rejected_ood".into();
            reason = Some("Out of distribution".into());
        } else if confidence < self.confidence_threshold {
            status = "rejected_low_confidence".into();
            reason = Some("Low confidence score".into());
        } else {
            params = self.ner.extract(input, &self.encoder)?;
        }

        Ok(AgentResult {
            input: input.to_string(),
            intent_id,
            confidence,
            ood_score,
            status,
            reason,
            parameters: params,
        })
    }
}
