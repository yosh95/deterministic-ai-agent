use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Linear, Optimizer, VarBuilder, VarMap, linear, ops};

pub struct IntentClassifierModel {
    fc1: Linear,
    fc2: Linear,
}

impl IntentClassifierModel {
    pub fn new(vs: VarBuilder, input_dim: usize, num_classes: usize) -> Result<Self> {
        let fc1 = linear(input_dim, 128, vs.pp("fc1"))?;
        let fc2 = linear(128, num_classes, vs.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }
}

pub struct NERClassifierModel {
    head: Linear,
}

impl NERClassifierModel {
    pub fn new(vs: VarBuilder, input_dim: usize, num_labels: usize) -> Result<Self> {
        let head = linear(input_dim, num_labels, vs.pp("ner_head"))?;
        Ok(Self { head })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.head.forward(x)
    }
}

pub struct Trainer {
    varmap: VarMap,
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            varmap: VarMap::new(),
        }
    }

    pub fn train_ner(
        &mut self,
        token_embeddings: &Tensor, // [N, Seq_Len, Hidden_Dim]
        labels: &Tensor,           // [N, Seq_Len]
        num_labels: usize,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let device = token_embeddings.device();
        let vs = VarBuilder::from_varmap(&self.varmap, DType::F32, device);
        let (n, seq_len, dim) = token_embeddings.dims3()?;

        let model = NERClassifierModel::new(vs, dim, num_labels)?;
        let mut opt = candle_nn::AdamW::new_lr(self.varmap.all_vars(), learning_rate)?;

        // Flatten for cross entropy: [N * Seq_Len, Dim]
        let x_flat = token_embeddings.reshape((n * seq_len, dim))?;
        let y_flat = labels.reshape(n * seq_len)?;

        for epoch in 1..=epochs {
            let logits = model.forward(&x_flat)?;
            let log_sm = ops::log_softmax(&logits, 1)?;
            let loss = candle_nn::loss::nll(&log_sm, &y_flat)?;

            opt.backward_step(&loss)?;

            if epoch % 20 == 0 || epoch == 1 {
                println!(
                    "NER Epoch: {:>3}, Loss: {:.4}",
                    epoch,
                    loss.to_vec0::<f32>()?
                );
            }
        }
        Ok(())
    }

    pub fn train_intent(
        &mut self,
        embeddings: &Tensor,
        labels: &Tensor,
        num_classes: usize,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let device = embeddings.device();
        let vs = VarBuilder::from_varmap(&self.varmap, DType::F32, device);
        let input_dim = embeddings.dims()[1];

        let model = IntentClassifierModel::new(vs, input_dim, num_classes)?;
        let mut opt = candle_nn::AdamW::new_lr(self.varmap.all_vars(), learning_rate)?;

        for epoch in 1..=epochs {
            let logits = model.forward(embeddings)?;
            let log_sm = ops::log_softmax(&logits, 1)?;
            let loss = candle_nn::loss::nll(&log_sm, labels)?;

            opt.backward_step(&loss)?;

            if epoch % 20 == 0 || epoch == 1 {
                println!("Epoch: {:>3}, Loss: {:.4}", epoch, loss.to_vec0::<f32>()?);
            }
        }
        Ok(())
    }

    pub fn save_weights(&self, path: &str) -> Result<()> {
        self.varmap.save(path)
    }

    pub fn get_centroids(
        &self,
        embeddings: &Tensor,
        labels_vec: &[u32],
        num_classes: usize,
    ) -> Result<Tensor> {
        let (_num_samples, dim) = embeddings.dims2()?;
        let mut centroid_data = Vec::with_capacity(num_classes * dim);

        for class_idx in 0..num_classes {
            let mut count = 0.0f32;
            let mut sum = vec![0.0f32; dim];

            for (i, &l) in labels_vec.iter().enumerate() {
                if l as usize == class_idx {
                    let row = embeddings.get(i)?.to_vec1::<f32>()?;
                    for (s, &r) in sum.iter_mut().zip(row.iter()) {
                        *s += r;
                    }
                    count += 1.0;
                }
            }

            if count > 0.0 {
                for s in sum.iter_mut() {
                    *s /= count;
                }
                // Normalize centroid
                let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-9;
                for s in sum.iter_mut() {
                    *s /= norm;
                }
                centroid_data.extend(sum);
            } else {
                centroid_data.extend(vec![0.0f32; dim]);
            }
        }

        Tensor::from_vec(centroid_data, (num_classes, dim), embeddings.device())
    }
}
