use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap, Optimizer};

pub struct IntentClassifier {
    fc: Linear,
    varmap: VarMap,
    centroids: Option<Tensor>,
}

impl IntentClassifier {
    pub fn new(input_dim: usize, num_intents: usize) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &candle_core::Device::Cpu);
        let fc = candle_nn::linear(input_dim, num_intents, vb.pp("fc"))?;
        Ok(Self {
            fc,
            varmap,
            centroids: None,
        })
    }

    /// Predict intent ID and confidence (max probability)
    pub fn predict_with_confidence(&self, x: &Tensor) -> Result<(u32, f32)> {
        // x shape is [dim], need [1, dim] for linear layer
        let x_batched = x.unsqueeze(0)?;
        let logits = self.fc.forward(&x_batched)?.squeeze(0)?;
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
        let logits = self.fc.forward(embeddings)?;
        
        // Log-softmax and NLL Loss
        let log_sm = candle_nn::ops::log_softmax(&logits, 1)?;
        let loss = candle_nn::loss::nll(&log_sm, labels)?;
        
        // Backward and Step
        opt.backward_step(&loss)?;

        loss.to_scalar::<f32>()
    }

    /// Update centroids for OOD detection using training data
    pub fn update_centroids(&mut self, embeddings: &Tensor, labels: &[u32]) -> Result<()> {
        let (_num_samples, dim) = embeddings.dims2()?;
        let num_classes = labels.iter().max().cloned().unwrap_or(0) as usize + 1;
        
        let mut class_sums = vec![vec![0.0f32; dim]; num_classes];
        let mut class_counts = vec![0usize; num_classes];
        
        let emb_data: Vec<Vec<f32>> = embeddings.to_vec2()?;
        for (i, label) in labels.iter().enumerate() {
            let label = *label as usize;
            for d in 0..dim {
                class_sums[label][d] += emb_data[i][d];
            }
            class_counts[label] += 1;
        }
        
        let mut centroid_data = Vec::with_capacity(num_classes * dim);
        for i in 0..num_classes {
            let count = class_counts[i] as f32;
            for d in 0..dim {
                centroid_data.push(if count > 0.0 { class_sums[i][d] / count } else { 0.0 });
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
            None => return Ok(1.0), // Default to 1.0 (in-distribution) if no centroids
        };

        // Normalize input: x is [384], x_norm is scalar
        let x_norm = x.sqr()?.sum_all()?.sqrt()?;
        let x_normalized = x.broadcast_div(&x_norm)?;

        // Normalize centroids: centroids is [num_classes, 384], c_norms is [num_classes]
        let c_norms = centroids.sqr()?.sum(1)?.sqrt()?.reshape(((), 1))?;
        let c_normalized = centroids.broadcast_div(&c_norms)?;

        // Matrix multiplication for cosine similarities: [num_classes, 384] * [384, 1] -> [num_classes]
        let similarities = c_normalized.matmul(&x_normalized.unsqueeze(1)?)?.squeeze(1)?;
        let max_sim = similarities.to_vec1()?.into_iter().fold(f32::MIN, f32::max);
        
        Ok(max_sim)
    }
}
