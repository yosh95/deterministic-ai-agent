use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

pub struct EmbeddingEncoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingEncoder {
    pub fn new(model_id: &str) -> Result<Self> {
        let device = Device::Cpu;

        // 1. Download files directly from Hugging Face Hub (No Python required)
        let api = Api::new()?;
        let repo = api.repo(Repo::model(model_id.to_string()));
        
        let config_path = repo.get("config.json")?;
        let weights_path = repo.get("model.safetensors")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        // 2. Load model and tokenizer
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)?
        };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Tensor> {
        // Tokenization
        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        let token_ids = tokens.get_ids();
        let token_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;

        // Forward pass to get embeddings
        let embeddings = self.model.forward(&token_ids_tensor, &token_type_ids, None)?;
        
        // Apply Mean Pooling to get a single vector for the sentence
        let (_n_batch, n_tokens, _hidden_size) = embeddings.dims3()?;
        let mean_embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        
        Ok(mean_embeddings.squeeze(0)?)
    }
}
