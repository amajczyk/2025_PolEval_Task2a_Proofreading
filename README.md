# Less is More: Achieving SOTA at PolEval 2025 Task 2a

**Gender-inclusive LLMs for Polish (Proofreading) with LoRA and Qwen3-8B**

[![Paper](https://img.shields.io/badge/Paper-ACL-blue)](https://github.com/amajczyk/2025_PolEval_Task2a_Proofreading)


## Results

This repository contains the **winning solution** for **PolEval 2025 Task 2a: Gender-inclusive LLMs for Polish (Proofreading)**. 

**Key Achievements:**
-  **New SOTA:** F1-Score of **0.6039** (competition submission)
-  **Post-competition improvement:** F1-Score of **0.6283±0.0056**
-  Outperformed previous SOTA (Bielik-11B, F1=0.5985) using a **smaller 8B model**
-  Efficient fine-tuning with **LoRA** on consumer-grade GPUs (RTX 4080/4090)

## Table of Contents

- [Overview](#overview)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Hardware Requirements](#hardware-requirements)
- [Acknowledgments](#acknowledgments)
- [Future Work](#future-work)
- [Contact](#contact)

## Overview

The Polish language encodes gender in parts of speech, creating challenges in professional and formal contexts through "generic masculine" forms that research shows evoke male mental connotations. This project addresses gender bias in Polish language generation by fine-tuning Large Language Models to produce gender-inclusive text.

Our solution demonstrates that **smaller models with efficient fine-tuning can outperform larger models**, achieving state-of-the-art results using:
- **Model:** Qwen3-8B (4-bit quantized)
- **Fine-tuning:** LoRA (Low-Rank Adaptation)
- **Framework:** Unsloth (optimized for low-VRAM GPUs)

## Task Description

**Goal:** Transform sequences containing masculine-biased forms into gender-inclusive versions while:
- Preserving original semantic meaning
- Maintaining grammatical structure
- Applying transformations only where contextually appropriate
- Distinguishing between generic and specific references

**Example:**
- **Input:** *"Nauczyciel powinien przygotować się do lekcji."*  
  ("The teacher should prepare for the lesson.")
- **Output:** *"Nauczyciel*ka powin*ien/na przygotować się do lekcji."*  
  (Gender-inclusive form using asterisk schema)

**Evaluation:** F1-score on normalized tokens after expanding all gender-inclusive schemas.

## Dataset

Based on the IPIS-proofreading dataset:

| Split | Examples | Avg Length (chars) |
|-------|----------|-------------------|
| Train | 23,532 | 466.37 (source) |
| Validation | 2,732 | 457.40 (source) |
| Test A | 2,639 | 586.57 (source) |
| Test B | 2,639 | 593.15 (source) |

**Key Statistics:**
- Average source text: ~466 characters
- Average target text: ~482 characters
- System prompt: 3,167 characters
- Total input: ~3,727 characters (7.74× longer than output)

## Methodology

### Model Architecture
- **Base Model:** Qwen3-8B-Instruct (4-bit quantized)
- **Fine-tuning:** LoRA with rank 64 (submitted), rank 32 (optimal post-competition)
- **Target Layers:** QKV, Output, Gate, Up, Down projections

### Training Configuration
```python
# LoRA Configuration
LoRA Rank: 64 (submitted) / 32 (best)
Applied Layers: QKV, Output, Gate, Up, Down

# Optimization
Epochs: 2
Optimizer: AdamW-8bit
Learning Rate: 2e-4
LR Scheduler: Cosine with 10 warmup steps
Weight Decay: 0.01

# Batching
Effective Batch Size: 2 (with gradient accumulation)
Max Sequence Length: 4096 tokens

# Loss Calculation
Target: Output tokens only (not input tokens)
```

### Inference Configuration
```python
Temperature: 0.3
Top-p: 0.9
Top-k: 50
Max New Tokens: 4096
```

### Key Innovation
**Loss calculation on output tokens only** — Unlike previous approaches, we calculate training loss exclusively on generated tokens rather than memorizing the lengthy system prompt. This prevents overfitting to instructions given the dataset's characteristic of short source texts vs. long prompts.

## Results

### Competition Results (Test B)

| Model | Parameters | Precision | Recall | F1 |
|-------|-----------|-----------|--------|-----|
| PLLuM-12B (Baseline) | 12B | 2.56 | 6.28 | 3.64 |
| Bielik-11B-tuned (Prev. SOTA) | 11B | 63.93 | 56.26 | 59.85 |
| **Qwen3-8B-LoRA (Ours, submitted)** | **8B** | **64.50** | **56.77** | **60.39** |

### Post-Competition Results

| LoRA Rank | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| r=8 | 0.9740±0.0007 | 0.6424±0.0136 | 0.5287±0.0042 | 0.5800±0.0074 |
| r=16 | 0.9740±0.0001 | 0.6460±0.0003 | 0.5143±0.0038 | 0.5727±0.0024 |
| **r=32** | **0.9764±0.0004** | **0.6670±0.0050** | **0.5940±0.0061** | **0.6283±0.0056** |
| r=64 | 0.9753±0.0008 | 0.6587±0.0121 | 0.5735±0.0083 | 0.6131±0.0094 |
| r=128 | 0.9754±0.0001 | 0.6540±0.0019 | 0.5671±0.0009 | 0.6074±0.0005 |

**Key Findings:**
- LoRA rank 32 provides the optimal balance (F1=0.6283±0.0056)
- Ranks 64+ show diminishing returns and potential overfitting
- Loss calculation on output tokens only slightly outperforms full-text loss
- 72-74% determinism across inference runs (temperature=0.3)

##  Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 4080 (16GB VRAM) or equivalent
- RAM: 32GB system memory
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA RTX 4090 (24GB VRAM) or better
- RAM: 64GB system memory
- Storage: 100GB free space (for multiple checkpoints)

## Acknowledgments

- PolEval 2025 organizers for hosting the shared task
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning framework
- Qwen team for the pre-trained language models
- IPIS dataset creators for the proofreading corpus

## Future Work

- Explore larger models (Qwen3-14B, Qwen3-32B) with similar approach
- Automated prompt engineering using optimization techniques
- Apply LoRA fine-tuning to Bielik-11B for direct comparison
- Expand to other Slavic languages with grammatical gender
- Investigate multi-task learning with translation (Task 2b)

## Contact

For questions, issues, or collaborations, please:
- Open an issue in this repository
- Contact: adam.majczyk@ipipan.waw.pl