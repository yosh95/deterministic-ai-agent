use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, api::sync::Api};
use tokenizers::Tokenizer;

pub struct EmbeddingEncoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingEncoder {
    pub fn new(model_id: &str) -> Result<Self> {
        let device = Device::Cpu;

        let api = Api::new()?;
        let repo = api.repo(Repo::model(model_id.to_string()));

        let config_path = repo.get("config.json")?;
        let weights_path = repo.get("model.safetensors")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let tensors = candle_core::safetensors::load(weights_path, &device)?;
        let vb = VarBuilder::from_tensors(tensors, DTYPE, &device);
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Encode full text into a single summary vector (Mean Pooling)
    /// E5 model requires "query: " prefix for tasks.
    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let prefixed = format!("query: {}", text);
        let (embeddings, mask) = self.get_hidden_states_with_mask(&prefixed)?;

        // Correct Mean Pooling: Only average non-padding tokens
        let mask_f32 = mask.to_dtype(DType::F32)?;
        let sum_embeddings = embeddings.broadcast_mul(&mask_f32.unsqueeze(1)?)?.sum(0)?;
        let sum_mask = mask_f32.sum_all()?.to_scalar::<f32>()? + 1e-9;
        let mean_pooled = (sum_embeddings / (sum_mask as f64))?;

        // Normalize for cosine similarity
        let norm = (mean_pooled.sqr()?.sum_all()?.sqrt()? + 1e-9)?;
        Ok(mean_pooled.broadcast_div(&norm)?)
    }

    pub fn get_hidden_states(&self, text: &str) -> Result<Tensor> {
        let (embs, _) = self.get_hidden_states_with_mask(text)?;
        Ok(embs)
    }

    pub fn get_hidden_states_with_mask(&self, text: &str) -> Result<(Tensor, Tensor)> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        Ok((embeddings.squeeze(0)?, attention_mask.squeeze(0)?))
    }

    pub fn get_tokens(&self, text: &str) -> Result<Vec<String>> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        Ok(tokens.get_tokens().to_vec())
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
