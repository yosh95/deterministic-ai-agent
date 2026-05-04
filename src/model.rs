use candle_core::{Result, Tensor, DType};
use candle_nn::{Linear, Module, VarBuilder, VarMap, Optimizer};
use std::collections::HashMap;

pub struct IntentClassifier {
    fc1: Linear,
    fc2: Linear,
    varmap: VarMap,
    centroids: Option<Tensor>,
}

pub struct TokenClassifier {
    head: Linear,
    varmap: VarMap,
}

impl TokenClassifier {
    pub fn new(input_dim: usize, num_labels: usize) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &candle_core::Device::Cpu);
        let head = candle_nn::linear(input_dim, num_labels, vb.pp("ner_head"))?;
        Ok(Self { head, varmap })
    }

    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.head.forward(x)
    }

    pub fn predict(&self, x: &Tensor) -> Result<Vec<u32>> {
        let logits = self.forward(x)?;
        let labels = logits.argmax(1)?;
        Ok(labels.to_vec1::<u32>()?)
    }
}

pub struct MasterMatcher {
    device_templates: HashMap<String, Tensor>, // Name -> Embedding
    threshold: f32,
}

impl MasterMatcher {
    pub fn new(threshold: f32) -> Self {
        Self {
            device_templates: HashMap::new(),
            threshold,
        }
    }

    pub fn add_template(&mut self, name: &str, embedding: Tensor) {
        self.device_templates.insert(name.to_string(), embedding);
    }

    pub fn match_entity(&self, embedding: &Tensor) -> Result<Option<(String, f32)>> {
        let mut best_name = None;
        let mut max_sim = -1.0f32;

        for (name, template) in &self.device_templates {
            let sim = self.cosine_similarity(embedding, template)?;
            if sim > max_sim && sim > self.threshold {
                max_sim = sim;
                best_name = Some((name.clone(), sim));
            }
        }
        Ok(best_name)
    }

    fn cosine_similarity(&self, v1: &Tensor, v2: &Tensor) -> Result<f32> {
        let dot = (v1 * v2)?.sum_all()?.to_scalar::<f32>()?;
        let norm1 = v1.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm2 = v2.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        Ok(dot / (norm1 * norm2 + 1e-7))
    }
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

    /// Load trained weights from a file
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    pub fn set_centroids(&mut self, centroids: Tensor) {
        self.centroids = Some(centroids);
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }

    /// Predict intent ID and confidence (max probability)
    pub fn predict_with_confidence(&self, x: &Tensor) -> Result<(u32, f32)> {
        let x_batched = x.unsqueeze(0)?;
        let logits = self.forward(&x_batched)?.squeeze(0)?;
        let probs = candle_nn::ops::softmax(&logits, 0)?;
        
        let mut max_prob = 0.0f32;
        let mut intent_id = 0u32;
        
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        for (i, &p) in probs_vec.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                intent_id = i as u32;
            }
        }
        
        Ok((intent_id, max_prob))
    }

    /// Train the classifier for one epoch
    pub fn train_one_epoch(&mut self, embeddings: &Tensor, labels: &Tensor, learning_rate: f64) -> Result<f32> {
        let mut opt = candle_nn::AdamW::new_lr(self.varmap.all_vars(), learning_rate)?;

        // Forward
        let logits = self.forward(embeddings)?;
        
        // Log-softmax and NLL Loss
        let log_sm = candle_nn::ops::log_softmax(&logits, 1)?;
        let loss = candle_nn::loss::nll(&log_sm, labels)?;
        
        // Backward and Step
        opt.backward_step(&loss)?;

        loss.to_scalar::<f32>()
    }

    /// Update centroids for OOD detection using training data
    pub fn update_centroids(&mut self, embeddings: &Tensor, labels: &[u32]) -> Result<()> {
        let (num_samples, dim) = embeddings.dims2()?;
        let num_classes = labels.iter().max().cloned().unwrap_or(0) as usize + 1;
        
        let mut centroid_data = Vec::with_capacity(num_classes * dim);
        
        for class_idx in 0..num_classes {
            let mut class_mask = Vec::with_capacity(num_samples);
            let mut count = 0.0f32;
            for &l in labels {
                if l as usize == class_idx {
                    class_mask.push(1.0f32);
                    count += 1.0;
                } else {
                    class_mask.push(0.0f32);
                }
            }
            
            if count > 0.0 {
                let mask_tensor = Tensor::from_vec(class_mask, (1, num_samples), embeddings.device())?;
                let class_emb_sum = mask_tensor.matmul(embeddings)?;
                let centroid = (class_emb_sum / (count as f64))?;
                centroid_data.extend(centroid.squeeze(0)?.to_vec1::<f32>()?);
            } else {
                centroid_data.extend(vec![0.0f32; dim]);
            }
        }
        
        let centroids = Tensor::from_vec(centroid_data, (num_classes, dim), embeddings.device())?;
        self.centroids = Some(centroids);
        Ok(())
    }

    /// Calculate OOD score based on cosine similarity to centroids
    pub fn get_ood_score(&self, x: &Tensor) -> Result<f32> {
        let centroids = match &self.centroids {
            Some(c) => c,
            None => return Ok(0.0), // Safe default: not in distribution
        };

        // Normalize input
        let x_norm = (x.sqr()?.sum_all()?.sqrt()? + 1e-8)?;
        let x_normalized = x.broadcast_div(&x_norm)?;

        // Normalize centroids
        let c_norms = (centroids.sqr()?.sum(1)?.sqrt()?.reshape(((), 1))? + 1e-8)?;
        let c_normalized = centroids.broadcast_div(&c_norms)?;

        // Max cosine similarity
        let similarities = c_normalized.matmul(&x_normalized.unsqueeze(1)?)?.squeeze(1)?;
        let max_sim = similarities.to_vec1()?.into_iter().fold(f32::MIN, f32::max);
        
        Ok(max_sim)
    }
}
