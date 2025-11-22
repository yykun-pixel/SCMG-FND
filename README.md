# SCMG-FND: Fake News Detection in Short Videos

A robust and interpretable multimodal framework for fake news detection in short videos, which integrates large-model–based semantic credibility assessment, online fact verification, and multi-granularity contrastive learning to enforce deep cross-modal consistency. Furthermore, the framework incorporates a neuro-symbolic reasoning module to enhance the transparency and reliability of predictions, enabling accurate identification of manipulated or misleading content across diverse and complex short-video scenarios.
## Project Title

**SCMG-FND**: Semantic Credibility and Multi-Granularity Fake News Detection

## Overview

Short videos have become a dominant medium for news delivery, but their low cost, rapid diffusion, and multimodal nature make misinformation easier to generate and harder to verify. SCMG-FND addresses these challenges by:

- Combining LLM-based video understanding with online search for multi-dimensional credibility assessment
- Employing multi-granularity contrastive learning to enforce cross-modal consistency
- Integrating neural-symbolic rule engine for interpretable decision-making
- Leveraging diffusion models for cross-modal viewpoint evolution

The framework achieves state-of-the-art performance on the FakeSV dataset with 89.11% accuracy and 89.53% F1 score, significantly outperforming existing methods.

## Features

- **Multi-dimensional Credibility Assessment**: Evaluates videos across nine dimensions (professionalism, factual consistency, AI-generation likelihood, editing artifacts, title-content consistency, emotional bias, misleading content, source reliability, and intent to spread)
- **Multi-granularity Contrastive Learning**: Enforces consistency at four levels (global, modal, temporal, and spatial)
- **Neural-Symbolic Rule Engine**: Provides interpretable logical constraints with soft matching and dynamic weighting
- **Cross-modal Fusion**: Diffusion-based viewpoint evolution and capsule network aggregation
- **Explainability Module**: Generates human-readable explanations with rule triggers and modality contributions
- **Robust Detection**: Handles emotional manipulation, title-content inconsistency, audio-video desynchronization, and local tampering

## Architecture

The framework consists of seven core components:

1. **Multimodal Feature Extraction**: Extracts features from transcript+title, comments, user profile, audio, key frames, and motion
2. **Multi-dimensional Credibility Assessment**: GLM-4V-Flash video understanding with online search for fact-checking
3. **Neural-Symbolic Rule Engine**: Soft matching in shared semantic space to calibrate features and reweight logits
4. **Cross-modal Interaction**: Multimodal Transformers for feature alignment
5. **Intra-modal Enhancement**: Capsule network for viewpoint aggregation
6. **Multi-granularity Contrastive Learning**: Unified InfoNCE objectives at global/modal/temporal/spatial levels
7. **Decision Fusion & Explainability**: MLP classifier with rule-based adjustments and explanation generation

## Repository Structure

```
SCMG-FND-mian/
├── config/                          # Configuration files
│   └── rule_structure_template.json
├── Diffusion/                       # Diffusion model modules
│   ├── ExplainableDetection.py
│   ├── Multimodal_Diffusion.py
│   └── Multimodal_Model.py
├── modules/                          # Core modules
│   ├── MultiGranularityContrast.py  # Multi-granularity contrastive learning
│   ├── NeuralSymbolicRules.py       # Neural-symbolic rule engine
│   ├── RobustExplainableFramework.py
│   └── transformer.py
├── src/                              # Source code
│   ├── CrossmodalTransformer.py
│   └── StoG.py
├── explanations/                     # Explainability modules
│   └── llm_enhanced_explainer.py
├── scripts/                          # Utility scripts
│   └── convert_docx_to_text.py
├── torchvggish-master/               # Audio feature extraction
├── main.py                          # Main training script
├── train.py                         # Training script
├── valid.py                         # Validation script
├── dataloader_fakesv.py             # Data loader for FakeSV dataset
├── config_neural_symbolic.py        # Neural-symbolic configuration
├── config_neural_symbolic_safe.py  # Safe configuration template
├── eval_metrics.py                  # Evaluation metrics
├── explanation_viewer.py            # Explanation visualization
├── extract_implicit_features.py     # Feature extraction
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites

- **GPU**: NVIDIA A100 (recommended) or compatible CUDA-enabled GPU
- **Python**: 3.8+
- **CUDA**: Compatible with PyTorch 2.1.0

### Step-by-step Installation

```bash
# Clone the repository
git clone <repository-url>
cd SCMG-FND-mian

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust CUDA version as needed)
# Visit https://pytorch.org/ for the correct installation command
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Key Dependencies

- `torch>=2.1.0`
- `transformers>=4.9.0`
- `numpy>=1.21.0`
- `opencv-python>=4.5.0`
- `scikit-learn>=0.24.0`

## Dataset Preparation

### FakeSV Dataset

The model is evaluated on the **FakeSV** dataset, a Chinese short-video fake news dataset:

- **Source**: TikTok and Kuaishou (2019-2022)
- **Total Videos**: 5,538 news videos
- **Distribution**: 
  - 1,827 fake cases (from 738 independent news events)
  - 1,827 true cases (verified by authoritative sources)
  - 1,884 debunking videos
- **Features**: Video frames, audio, transcripts, comments, user profiles
- **Split**: 70% training, 20% validation, 10% testing

### Data Format

The dataset should be organized as follows:

```
data/
├── videos/              # Video files
├── audio/               # Audio files
├── frames/              # Extracted key frames
├── transcripts/         # Text transcripts
├── comments/            # User comments
└── metadata.json        # Video metadata and labels
```

### Preprocessing

```bash
# Extract features (if needed)
python extract_implicit_features.py --data_path <path_to_data>
```

## Training

### Basic Training

```bash
# Train with default configuration
python train.py --config config_neural_symbolic.py

# Train with custom parameters
python train.py \
    --config config_neural_symbolic.py \
    --batch_size 32 \
    --learning_rate 0.00005 \
    --epochs 60
```

### Training Parameters

- **Learning Rate**: 0.00005 (with cosine annealing)
- **Batch Size**: 32
- **Epochs**: 60 (with early stopping)
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, weight decay=0.01)
- **Mixed Precision**: Enabled for efficiency
- **Gradient Clipping**: Max norm 1.0

### Training Output

Training logs and checkpoints are saved to:
- `logs/`: TensorBoard logs
- `checkpoints/`: Model checkpoints

## Evaluation

### Validation

```bash
# Evaluate on validation set
python valid.py --config config_neural_symbolic.py --checkpoint <path_to_checkpoint>
```

### Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity

### Expected Results

On FakeSV test set:
- **Accuracy**: 89.11% ± 0.04%
- **F1 Score**: 89.53% ± 0.05%
- **Precision**: 90.73%
- **Recall**: 88.27%

## Inference

### Single Video Detection

```bash
# Run inference on a single video
python main.py \
    --mode inference \
    --video_path <path_to_video> \
    --checkpoint <path_to_checkpoint> \
    --config config_neural_symbolic.py
```

### Batch Inference

```bash
# Process multiple videos
python main.py \
    --mode batch_inference \
    --data_path <path_to_video_directory> \
    --checkpoint <path_to_checkpoint> \
    --config config_neural_symbolic.py \
    --output_path <output_directory>
```

### Explanation Generation

```bash
# Generate explanations for predictions
python explanation_viewer.py \
    --video_path <path_to_video> \
    --checkpoint <path_to_checkpoint> \
    --output_path <explanation_output>
```

## Configuration Files

### Main Configuration

- **`config_neural_symbolic.py`**: Main configuration file with neural-symbolic rules
- **`config_neural_symbolic_safe.py`**: Safe template (sensitive parameters removed)

### Configuration Parameters

Key parameters in configuration files:

```python
# Training parameters
learning_rate = 0.00005
batch_size = 32
epochs = 60

# Model parameters
hidden_dim = 768
num_heads = 12
num_layers = 6

# Contrastive learning
temperature = 0.1
lambda_global = 1.0
lambda_modal = 1.0
lambda_temporal = 1.0
lambda_spatial = 1.0

# Neural-symbolic rules
rule_threshold = 0.75
enable_neural_symbolic = True
```

### Environment Variables

For sensitive parameters, use environment variables:

```bash
export NEURAL_SYMBOLIC_WEIGHT=0.5
export RULE_THRESHOLD=0.75
export RULE_CONFIDENCE_THRESHOLD=0.8
```

## Pretrained Models (Optional)

Pretrained models (if available) should be placed in `checkpoints/` directory.

```bash
# Download pretrained model (if available)
# Place in checkpoints/ directory

# Load pretrained model
python valid.py \
    --config config_neural_symbolic.py \
    --checkpoint checkpoints/pretrained_model.pth
```

## Security & Privacy Notice

### Data Privacy

- The FakeSV dataset is used solely for academic research
- No misleading content is disseminated
- User data is handled according to relevant privacy regulations

### Model Security

- Sensitive parameters (weights, thresholds) are not hardcoded
- Use `config_neural_symbolic_safe.py` for deployment
- LLM-generated analyses are not shared publicly to prevent hallucination propagation

### Ethical Considerations

- **Dual-use Risk**: The model could potentially be reverse-engineered to create more deceptive content
- **Mitigation**: 
  - Model parameters and internal thresholds are not disclosed
  - Modular design increases reverse-engineering cost
  - Strict functional interface limitations

See `SECURITY.md` for detailed security guidelines.

## Cross-Lingual Support (Optional)

Currently optimized for **Chinese** short videos. For other languages:

1. Replace Chinese language models (RoBERTa, GLM-4V-Flash) with language-specific models
2. Update tokenizers and vocabulary
3. Retrain on language-specific datasets
4. Adjust prompt templates for LLM-based credibility assessment


## Limitations

1. **LLM Dependence**: Relies on GLM-4V-Flash and Kimi for credibility assessment. Performance may vary with LLM capabilities and knowledge coverage, especially for domain-specific content.

2. **Computational Complexity**: 
   - Training: ~3 minutes per epoch on A100 (80GB)
   - Inference: ~0.3s per video on A100, ~2.8s on standard servers
   - With cloud APIs: ~2 minutes per video (non-parallelized)

3. **Dataset Dependency**: 
   - Optimized for Chinese short videos (FakeSV dataset, pre-2023)
   - Limited cross-lingual transferability
   - May not cover emerging forgery types (AI-generated avatars, real-time manipulation)

4. **External API Dependency**: Online search for fact-checking requires internet connectivity and API access.



## License

This project is for academic research purposes only. See LICENSE file for details.

## Acknowledgements

- This research was funded by National Natural Science Foundation of China, grant number U24B20147
- We thank the anonymous reviewers for their valuable comments and suggestions
- The FakeSV dataset creators for providing the benchmark dataset

## Contact / Issues

- **Corresponding Author**: Yijia Xu (xuyijia@scu.edu.cn)
- **Institution**: School of Cyber Science and Engineering, Sichuan University, Chengdu 610207, China
- **Issues**: Please open an issue on GitHub for bug reports or feature requests
- **Questions**: For research-related questions, please contact the corresponding author

---

**Note**: This framework is designed for content authenticity verification, not for generating or enhancing potentially misleading content. Please use responsibly.
