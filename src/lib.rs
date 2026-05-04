pub mod encoder;
pub mod model;
pub mod ner;
pub mod train;

use crate::encoder::EmbeddingEncoder;
use crate::model::IntentClassifier;
use crate::ner::ModelNER;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("Config error: {0}")]
    ConfigError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Anyhow error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;

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
    pub in_dist_similarity: f32,
    pub status: String,
    pub reason: Option<String>,
    pub parameters: HashMap<String, String>,
}

pub struct AgentEngine {
    encoder: Arc<EmbeddingEncoder>,
    classifier: IntentClassifier,
    ner: ModelNER,
    confidence_threshold: f32,
    ood_threshold: f32, // Similarity threshold (higher is more restrictive)
}

impl AgentEngine {
    pub fn new(
        encoder: Arc<EmbeddingEncoder>,
        classifier: IntentClassifier,
        ner: ModelNER,
        config_path: Option<&str>,
    ) -> Result<Self> {
        let mut confidence_threshold = 0.70;
        let mut ood_threshold = 0.88;

        if let Some(path) = config_path {
            let content = std::fs::read_to_string(path)
                .map_err(|e| AgentError::ConfigError(format!("Failed to read config: {}", e)))?;
            let config: EngineConfig = serde_yaml::from_str(&content)
                .map_err(|e| AgentError::ConfigError(format!("Failed to parse YAML: {}", e)))?;

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
        let in_dist_similarity = self.classifier.get_in_dist_similarity(&vector)?;

        let mut status = "success".to_string();
        let mut reason = None;
        let mut params = HashMap::new();

        if in_dist_similarity < self.ood_threshold {
            status = "rejected_ood".into();
            reason = Some("Input out of distribution".into());
        } else if confidence < self.confidence_threshold {
            status = "rejected_low_confidence".into();
            reason = Some("Low intent confidence".into());
        } else {
            params = self.ner.extract(input, &self.encoder)?;
        }

        Ok(AgentResult {
            input: input.to_string(),
            intent_id,
            confidence,
            in_dist_similarity,
            status,
            reason,
            parameters: params,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::EmbeddingEncoder;
    use crate::model::{IntentClassifier, MasterMatcher, NERClassifier};

    #[test]
    fn test_engine_rejection_logic() -> Result<()> {
        // Mock components (using small model for speed in real env, here we focus on logic)
        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        let encoder = EmbeddingEncoder::new(model_id)?;
        let classifier = IntentClassifier::new(384, 2)?;
        let ner = ModelNER::new(384, 0.8)?;

        let mut engine = AgentEngine::new(encoder, classifier, ner, None)?;

        // Setup very strict thresholds
        engine.ood_threshold = 1.0; // Impossible similarity
        let res = engine.run_step("test input")?;
        assert_eq!(res.status, "rejected_ood");

        // Setup very low OOD but strict confidence
        engine.ood_threshold = 0.0;
        engine.confidence_threshold = 1.0; // Impossible confidence
        let res = engine.run_step("test input")?;
        assert_eq!(res.status, "rejected_low_confidence");

        Ok(())
    }
}
