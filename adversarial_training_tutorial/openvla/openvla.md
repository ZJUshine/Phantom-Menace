# OpenVLA Adversarial Training Tutorial

This guide provides step-by-step instructions for conducting adversarial training on the OpenVLA model using sensor attack augmentation.

## Prerequisites

### 1. Environment Setup

Install the OpenVLA environment following the instructions in the [official OpenVLA GitHub repository](https://github.com/openvla/openvla).

### 2. Dataset Preparation

Download the modified LIBERO dataset and place it in your `openvla` directory:

- **Dataset**: [openvla/modified_libero_rlds on Hugging Face](https://huggingface.co/datasets/openvla/modified_libero_rlds)

## Configuration Steps

### 3. Add Adversarial Training Script

Copy the adversarial training script to the VLA scripts directory:

- **File**: `adversarial_finetune.py`
- **Destination**: `openvla/vla-scripts/`

### 4. Import Sensor Attack Scripts

Place the sensor attack implementation in the root directory:

- **File Dir**: `sensor_attacks`
- **Destination**: Root directory of OpenVLA

### 5. Modify Dataset Loading

Integrate sensor attacks into the dataset pipeline by modifying the following files:

**Step 5.1**: Update `openvla/prismatic/vla/datasets/datasets.py`

- Reference the modified `datasets.py` file provided

**Step 5.2**: Update `openvla/prismatic/vla/datasets/__init__.py`

Add the adversarial batch transform to the imports:

```python
from .datasets import DummyDataset, EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, AdversarialRLDSBatchTransform
```

## Training

### 6. Launch Adversarial Training

Run the following command to start adversarial training with LoRA fine-tuning:

```bash
export WANDB_BASE_URL=https://api.wandb-cn.top

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/adversarial_finetune.py \
  --vla_path "openvla/openvla-7b" \
  --adversarial_ratio 0.3 \
  --data_root_dir /root/autodl-tmp/openvla/modified_libero_rlds \
  --dataset_name libero_object_no_noops \
  --run_root_dir /root/autodl-tmp/openvla/ \
  --adapter_tmp_dir /root/autodl-tmp/openvla/adapter_tmp_dir \
  --lora_rank 32 \
  --batch_size 12 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla-adversarial-lora-libero-object \
  --wandb_entity zjushine \
  --save_steps 2000
```

### Hardware Requirements

- `batch_size=12` requires **48GB VRAM**
- `batch_size=24` requires **80GB VRAM**

## Key Parameters

- `adversarial_ratio`: Proportion of adversarial examples in each batch (0.3 = 30%)
- `lora_rank`: Rank of LoRA adapter matrices
- `image_aug`: Enable/disable image augmentation
- `save_steps`: Checkpoint saving frequency