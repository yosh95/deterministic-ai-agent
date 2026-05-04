use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct DeviceConfig {
    #[allow(dead_code)]
    devices: Vec<String>,
    prefixes: Vec<String>,
}

pub struct NERExtractor {
    #[allow(dead_code)]
    devices: Vec<String>,
    device_regex: Regex,
    sensor_regex: Regex,
    line_regex: Regex,
    numeric_regex: Regex,
    fault_regex: Regex,
}

impl NERExtractor {
    pub fn new(config_path: &str) -> Result<Self> {
        let config_content = std::fs::read_to_string(config_path).unwrap_or_else(|_| "devices: []\nprefixes: []".into());
        let config: DeviceConfig = serde_yaml::from_str(&config_content)?;

        let mut patterns = Vec::new();
        if !config.devices.is_empty() {
            let escaped_devices: Vec<String> = config.devices.iter().map(|d| regex::escape(d)).collect();
            patterns.push(format!("({})", escaped_devices.join("|")));
        }
        if !config.prefixes.is_empty() {
            let escaped_prefixes: Vec<String> = config.prefixes.iter().map(|p| regex::escape(p)).collect();
            patterns.push(format!("(({})[A-Za-z0-9_]+)", escaped_prefixes.join("|")));
        }

        let device_pattern = if patterns.is_empty() {
            r"\b(Unknown_Device)\b".to_string()
        } else {
            format!(r"\b({})\b", patterns.join("|"))
        };

        Ok(Self {
            devices: config.devices,
            device_regex: Regex::new(&format!("(?i){}", device_pattern))?,
            sensor_regex: Regex::new(r"(?i)\bsensor\s+([A-Za-z0-9_\-]+)\b")?,
            line_regex: Regex::new(r"(?i)\b(?:line|ライン)\s*(\d+)\b")?,
            numeric_regex: Regex::new(r"\b(\d+(?:\.\d+)?)\s*(?:°C|bar|rpm|kPa|V|A|Hz)?\b")?,
            fault_regex: Regex::new(r"(?i)\b(overheat|overvoltage|overcurrent|vibration|calibration\s+drift|pressure\s+drop|manual\s?stop|shutdown)\b")?,
        })
    }

    pub fn extract(&self, text: &str) -> HashMap<String, String> {
        let mut params = HashMap::new();

        if let Some(cap) = self.device_regex.captures(text) {
            params.insert("device_id".to_string(), cap[1].to_string());
            params.insert("extraction_method".to_string(), "regex".to_string());
        }

        if let Some(cap) = self.sensor_regex.captures(text) {
            params.insert("sensor".to_string(), cap[1].to_uppercase());
        }

        if let Some(cap) = self.line_regex.captures(text) {
            params.insert("line_id".to_string(), cap[1].to_string());
        }

        if let Some(cap) = self.numeric_regex.captures(text) {
            params.insert("value".to_string(), cap[1].to_string());
        }

        if let Some(cap) = self.fault_regex.captures(text) {
            let fault = cap[1].to_lowercase().replace(' ', "_").replace('-', "_");
            params.insert("fault".to_string(), fault);
        }

        params
    }
}
