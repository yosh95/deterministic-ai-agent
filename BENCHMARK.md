# Performance & Environment Report: Deterministic AI Agent

## 1. Executive Summary
本プロジェクトでは、OT（運用技術）環境に最適化された「ハイブリッドONNXシステム」の開発に成功した。
学習にはPyTorchの柔軟性を活かし、推論にはONNX Runtime + Numpyを用いることで、**「Zero-Torch Inference」**（推論時にPyTorchをロードしない構成）を実現。メモリ消費の大幅な削減と、3倍以上の高速化を達成した。

### 三層防御モデル (Triple-Guard Model)
- **L1 (Determinism):** `argmax` による決定論的アクション選択。
- **L2 (Confidence Gate):** ソフトマックス閾値（デフォルト: 0.70）による曖昧な指示の拒絶。
- **L3 (OOD Detection):** 重心ベース余弦類似度（デフォルト: 0.88）によるドメイン外入力（雑談等）の排除。

## 2. Infrastructure & Hardware Environment
本ベンチマークは、Windows 11上のWSL2環境で測定された。

### Host Machine (Windows)
| 項目 | 詳細 |
| :--- | :--- |
| **システムモデル** | Dell Inspiron 14 5445 |
| **Host OS** | Microsoft Windows 11 Home (Build 26200) |
| **プロセッサ** | AMD Ryzen 5 8540U (6 Cores, 12 Logical Processors) @ 3201 Mhz |
| **物理メモリ (RAM)** | 16.0 GB |

### Execution Environment (Linux/WSL2)
| 項目 | 詳細 |
| :--- | :--- |
| **Execution OS** | **Ubuntu 24.04 LTS (WSL2)** |
| **Python Version** | 3.12.3 |
| **Runtime Backend** | **ONNX Runtime (CPU)** + **Numpy** |

## 3. Verified Performance Gains (PyTorch vs ONNX)
`benchmarks/profiler.py` および `benchmarks/v2_perf_comparison.py` による実測値。

| メトリクス | PyTorch (Baseline) | **ONNX (Measured)** | 改善率 |
| :--- | :--- | :--- | :--- |
| **コールドスタート時間** | 9.29s | **2.87s** | **3.2倍 高速** |
| **平均推論レイテンシ** | 25.36ms | **8.08ms** | **3.1倍 高速** |
| **推定スループット** | ~39 ops/sec | **~123 ops/sec** | **3.1倍 向上** |
| **依存ライブラリ** | 重厚 (PyTorch 700MB+) | **軽量 (Numpy/ORT)** | **RAM占有極小** |

## 4. Safety Gate Verification (L2/L3 Results)
OT特化データセット（500サンプル）によるドメイン適応後の挙動。

| 入力内容 | 分類 | OOD Score (L3) | 判定 | 理由 |
| :--- | :--- | :---: | :---: | :--- |
| `Conveyor_A vibration detected.` | 産業ドメイン | 0.98 | **実行** | 高信頼度 & In-Domain |
| `Status of Motor_B?` | 産業ドメイン | 0.96 | **実行** | 高信頼度 & In-Domain |
| `What is the meaning of life?` | ドメイン外 (OOD) | 0.42 | **拒絶** | OOD検出 (L3) |
| `Tell me a joke.` | ドメイン外 (OOD) | 0.38 | **拒絶** | OOD検出 (L3) |

## 5. Software Quality & Compliance
- **Zero-Torch Inference:** 推論パスにおいて `import torch` を一切行わない。
- **Typing:** `mypy` 準拠（ソースコード全17ファイル）。
- **Testing:** `pytest` 98項目すべて合格（静的解析・型チェック含む）。
- **NER Refinement:** `config/devices.yaml` 連携によるデバイスIDの正規表現抽出を実装。

---
*Report Updated: 2026-03-27*
