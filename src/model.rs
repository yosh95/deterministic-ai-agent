use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};

/// Intent classification neural network
pub struct IntentClassifier {
    fc1: Linear,
    fc2: Linear,
    varmap: VarMap,
    centroids: Option<Tensor>,
}

impl IntentClassifier {
    pub fn new(input_dim: usize, num_intents: usize) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &candle_core::Device::Cpu);
        let fc1 = candle_nn::linear(input_dim, 128, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(128, num_intents, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            varmap,
            centroids: None,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }

    pub fn predict_with_confidence(&self, x: &Tensor) -> Result<(u32, f32)> {
        let x_batched = x.unsqueeze(0)?;
        let logits = self.forward(&x_batched)?.squeeze(0)?;
        let probs = candle_nn::ops::softmax(&logits, 0)?;

        let probs_vec: Vec<f32> = probs.to_vec1()?;
        let (intent_id, confidence) = probs_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, v)| (i as u32, *v))
            .unwrap_or((0, 0.0));

        Ok((intent_id, confidence))
    }

    /// Calculate similarity to class centroids for OOD detection
    pub fn get_in_dist_similarity(&self, x: &Tensor) -> Result<f32> {
        let centroids = match &self.centroids {
            Some(c) => c,
            None => return Ok(0.0),
        };
        let similarities = centroids.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        let max_sim = similarities.to_vec1()?.into_iter().fold(f32::MIN, f32::max);
        Ok(max_sim)
    }

    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }
    pub fn set_centroids(&mut self, centroids: Tensor) {
        self.centroids = Some(centroids);
    }
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)
    }
}

/// Token classification (NER) neural network
pub struct NERClassifier {
    head: Linear,
    varmap: VarMap,
}

impl NERClassifier {
    pub fn new(input_dim: usize, num_labels: usize) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &candle_core::Device::Cpu);
        let head = candle_nn::linear(input_dim, num_labels, vb.pp("ner_head"))?;
        Ok(Self { head, varmap })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.head.forward(x)
    }

    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)
    }
}

/// Deterministic matching engine (Exact match + Embedding similarity)
pub struct MasterMatcher {
    device_templates: std::collections::HashMap<String, Tensor>,
    threshold: f32,
}

impl MasterMatcher {
    pub fn new(threshold: f32) -> Self {
        Self {
            device_templates: std::collections::HashMap::new(),
            threshold,
        }
    }

    pub fn add_template(&mut self, name: &str, embedding: Tensor) {
        self.device_templates.insert(name.to_string(), embedding);
    }

    pub fn match_entity(
        &self,
        token_str: &str,
        embedding: &Tensor,
    ) -> Result<Option<(String, f32)>> {
        // 1. Exact match (Deterministic priority)
        let clean_token = token_str.replace("##", "");
        for name in self.device_templates.keys() {
            if name.eq_ignore_ascii_case(&clean_token) {
                return Ok(Some((name.clone(), 1.0)));
            }
        }

        // 2. Embedding similarity
        let mut best_name = None;
        let mut max_sim = -1.0f32;
        let norm = (embedding.sqr()?.sum_all()?.sqrt()? + 1e-9)?;
        let norm_emb = embedding.broadcast_div(&norm)?;

        for (name, template) in &self.device_templates {
            let sim = norm_emb
                .broadcast_mul(template)?
                .sum_all()?
                .to_scalar::<f32>()?;
            if sim > max_sim && sim > self.threshold {
                max_sim = sim;
                best_name = Some((name.clone(), sim));
            }
        }
        Ok(best_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_master_matcher_exact_vs_similarity() -> Result<()> {
        let mut matcher = MasterMatcher::new(0.8);
        let device = Device::Cpu;

        let name = "Motor_A";
        // Create a fake normalized template (1.0 at index 0)
        let mut data = vec![0.0f32; 384];
        data[0] = 1.0;
        let template = Tensor::from_vec(data, (384,), &device)?;
        matcher.add_template(name, template);

        // 1. Test Exact Match (even if embedding is zero/random)
        let random_emb = Tensor::zeros((384,), DType::F32, &device)?;
        let res = matcher.match_entity("Motor_A", &random_emb)?;
        assert!(res.is_some());
        assert_eq!(res.unwrap().0, "Motor_A");

        // 2. Test Similarity Match (not exact string, but similar embedding)
        let mut sim_data = vec![0.0f32; 384];
        sim_data[0] = 0.95; // High similarity to template
        let sim_emb = Tensor::from_vec(sim_data, (384,), &device)?;
        let res = matcher.match_entity("Unknown_Device", &sim_emb)?;
        assert!(res.is_some());
        assert_eq!(res.unwrap().0, "Motor_A");

        // 3. Test No Match (low similarity)
        let mut low_sim_data = vec![0.0f32; 384];
        low_sim_data[10] = 1.0;
        let low_sim_emb = Tensor::from_vec(low_sim_data, (384,), &device)?;
        let res = matcher.match_entity("Unknown", &low_sim_emb)?;
        assert!(res.is_none());

        Ok(())
    }
}
