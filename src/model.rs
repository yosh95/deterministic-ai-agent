use candle_core::{DType, Result, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::collections::HashMap;

pub struct IntentClassifier {
    fc1: Linear,
    fc2: Linear,
    varmap: VarMap,
    optimizer: Option<AdamW>,
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
            optimizer: None,
            centroids: None,
        })
    }

    pub fn train_one_epoch(
        &mut self,
        embeddings: &Tensor,
        labels: &Tensor,
        learning_rate: f64,
    ) -> Result<f32> {
        if self.optimizer.is_none() {
            let params = ParamsAdamW {
                lr: learning_rate,
                ..Default::default()
            };
            let opt = AdamW::new(self.varmap.all_vars(), params)?;
            self.optimizer = Some(opt);
        }

        let logits = self.forward(embeddings)?;
        let log_sm = candle_nn::ops::log_softmax(&logits, 1)?;
        let loss = candle_nn::loss::nll(&log_sm, labels)?;

        if let Some(opt) = &mut self.optimizer {
            opt.backward_step(&loss)?;
        }

        loss.to_scalar::<f32>()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

    pub fn get_ood_score(&self, x: &Tensor) -> Result<f32> {
        let centroids = match &self.centroids {
            Some(c) => c,
            None => return Ok(0.0),
        };

        // x is assumed already normalized by encoder
        let similarities = centroids.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        let max_sim = similarities.to_vec1()?.into_iter().fold(f32::MIN, f32::max);

        Ok(max_sim)
    }

    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    pub fn set_centroids(&mut self, centroids: Tensor) {
        self.centroids = Some(centroids);
    }

    pub fn update_centroids(&mut self, embeddings: &Tensor, labels: &[u32]) -> Result<()> {
        let (_num_samples, dim) = embeddings.dims2()?;
        let num_classes = labels.iter().max().cloned().unwrap_or(0) as usize + 1;

        let mut centroid_data = Vec::with_capacity(num_classes * dim);

        for class_idx in 0..num_classes {
            let mut class_sum = vec![0.0f32; dim];
            let mut count = 0.0f32;
            for (i, &l) in labels.iter().enumerate() {
                if l as usize == class_idx {
                    let vec = embeddings.get(i)?.to_vec1::<f32>()?;
                    for (s, &v) in class_sum.iter_mut().zip(vec.iter()) {
                        *s += v;
                    }
                    count += 1.0;
                }
            }

            if count > 0.0 {
                for s in class_sum.iter_mut() {
                    *s /= count;
                }
                // Normalize centroid for cosine similarity
                let norm = class_sum.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-9;
                for s in class_sum.iter_mut() {
                    *s /= norm;
                }
            }
            centroid_data.extend(class_sum);
        }

        let centroids = Tensor::from_vec(centroid_data, (num_classes, dim), embeddings.device())?;
        self.centroids = Some(centroids);
        Ok(())
    }
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
}

pub struct MasterMatcher {
    device_templates: HashMap<String, Tensor>, // Name -> Normalized Embedding
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

        // Ensure token embedding is normalized
        let norm = (embedding.sqr()?.sum_all()?.sqrt()? + 1e-9)?;
        let norm_emb = embedding.broadcast_div(&norm)?;

        for (name, template) in &self.device_templates {
            let sim = norm_emb.broadcast_mul(template)?
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
