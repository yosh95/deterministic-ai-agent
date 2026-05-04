use crate::model::{IntentClassifier, NERClassifier};
use candle_core::{Result, Tensor};
use candle_nn::{Optimizer, ops};

pub struct Trainer;

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {
    pub fn new() -> Self {
        Self
    }

    /// Train the NER classifier
    pub fn train_ner(
        &self,
        classifier: &mut NERClassifier,
        token_embeddings: &Tensor, // [N, Seq_Len, Hidden_Dim]
        labels: &Tensor,           // [N, Seq_Len]
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let (n, seq_len, dim) = token_embeddings.dims3()?;
        let mut opt = candle_nn::AdamW::new_lr(classifier.varmap().all_vars(), learning_rate)?;

        let x_flat = token_embeddings.reshape((n * seq_len, dim))?;
        let y_flat = labels.reshape(n * seq_len)?;

        for epoch in 1..=epochs {
            let logits = classifier.forward(&x_flat)?;
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

    /// Train the Intent classifier
    pub fn train_intent(
        &self,
        classifier: &mut IntentClassifier,
        embeddings: &Tensor,
        labels: &Tensor,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let mut opt = candle_nn::AdamW::new_lr(classifier.varmap().all_vars(), learning_rate)?;

        for epoch in 1..=epochs {
            let logits = classifier.forward(embeddings)?;
            let log_sm = ops::log_softmax(&logits, 1)?;
            let loss = candle_nn::loss::nll(&log_sm, labels)?;
            opt.backward_step(&loss)?;

            if epoch % 20 == 0 || epoch == 1 {
                println!(
                    "Intent Epoch: {:>3}, Loss: {:.4}",
                    epoch,
                    loss.to_vec0::<f32>()?
                );
            }
        }
        Ok(())
    }

    /// Calculate class centroids in O(n) for OOD detection
    pub fn calculate_centroids(
        &self,
        embeddings: &Tensor,
        labels_vec: &[u32],
        num_classes: usize,
    ) -> Result<Tensor> {
        let (_num_samples, dim) = embeddings.dims2()?;
        let mut sums = vec![vec![0.0f32; dim]; num_classes];
        let mut counts = vec![0.0f32; num_classes];

        let emb_data = embeddings.to_vec2::<f32>()?;
        for (i, &label) in labels_vec.iter().enumerate() {
            let l = label as usize;
            if l < num_classes {
                for (j, &val) in emb_data[i].iter().enumerate() {
                    sums[l][j] += val;
                }
                counts[l] += 1.0;
            }
        }

        let mut centroid_data = Vec::with_capacity(num_classes * dim);
        for i in 0..num_classes {
            let mut row = sums[i].clone();
            if counts[i] > 0.0 {
                for val in row.iter_mut() {
                    *val /= counts[i];
                }
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-9;
                for val in row.iter_mut() {
                    *val /= norm;
                }
            }
            centroid_data.extend(row);
        }

        Tensor::from_vec(centroid_data, (num_classes, dim), embeddings.device())
    }
}
