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
    max_len: usize,
}

impl EmbeddingEncoder {
    pub fn new(model_id: &str) -> Result<Self> {
        let device = Device::Cpu;
        let max_len = 512;

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
            max_len,
        })
    }

    /// Encode full text into a single summary vector (Mean Pooling)
    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let embeddings = self.get_hidden_states(text)?;
        
        // Sum across tokens and divide by count (ignoring mask for simplicity here, 
        // but recommendation is to keep the mask logic if consistency is critical)
        let mean_embeddings = embeddings.mean(0)?;
        Ok(mean_embeddings)
    }

    /// Get raw hidden states for each token for NER/Token Classification
    pub fn get_hidden_states(&self, text: &str) -> Result<Tensor> {
        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        
        let token_ids = tokens.get_ids();
        let token_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;

        let embeddings = self.model.forward(&token_ids_tensor, &token_type_ids, None)?;
        Ok(embeddings.squeeze(0)?)
    }

    pub fn get_tokens(&self, text: &str) -> Result<Vec<String>> {
        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        Ok(tokens.get_tokens().to_vec())
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
