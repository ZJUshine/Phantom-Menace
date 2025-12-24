# OpenVLA-OFT Adversarial Training Tutorial

This guide provides step-by-step instructions for conducting adversarial training on the OpenVLA-OFT (Orthogonal Fine-Tuning) variant with sensor attack augmentation.

## Prerequisites

### 1. Environment Setup

Install the OpenVLA-OFT environment following the instructions in the [official OpenVLA-OFT GitHub repository](https://github.com/openvla/openvla).

### 2. Dataset Preparation

Download the modified LIBERO dataset and place it in your `openvla-oft` directory:

- **Dataset**: [openvla/modified_libero_rlds on Hugging Face](https://huggingface.co/datasets/openvla/modified_libero_rlds)

## Configuration Steps

### 3. Add Adversarial Training Script

Copy the adversarial training script to the VLA scripts directory:

- **File**: `adversarial_finetune.py`
- **Destination**: `openvla-oft/vla-scripts/`

### 4. Import Sensor Attack Scripts

Place the sensor attack implementation in the root directory:

- **File Dir**: `sensor_attacks`
- **Destination**: Root directory of OpenVLA-OFT

### 5. Modify Dataset Loading

Integrate sensor attacks into the dataset pipeline by modifying the following files:

**Step 5.1**: Update `openvla-oft/prismatic/vla/datasets/datasets.py`

- Reference the modified `datasets.py` file provided

**Step 5.2**: Update `openvla-oft/prismatic/vla/datasets/__init__.py`

Add the adversarial batch transform to the imports:

```python
from .datasets import DummyDataset, EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, AdversarialRLDSBatchTransform
```

## Training

### 6. Launch Adversarial Training

Run the following command to start adversarial training with the OFT variant:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_BASE_URL=https://api.wandb-cn.top

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/adversarial_finetune.py \
  --vla_path openvla/openvla-7b \
  --adversarial_ratio 0.3 \
  --data_root_dir /root/autodl-tmp/openvla-oft/modified_libero_rlds \
  --dataset_name libero_object_no_noops \
  --run_root_dir /root/autodl-tmp/openvla-oft/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "zjushine" \
  --wandb_project "openvla-oft-adversarial-lora-libero-object" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```

## Key Parameters

### Model Configuration
- `vla_path`: Path to the base OpenVLA model checkpoint
- `use_l1_regression`: Enable L1 regression loss (recommended: True)
- `use_diffusion`: Enable diffusion-based action prediction (set to False for standard regression)
- `use_film`: Enable FiLM conditioning layers

### Input Configuration
- `num_images_in_input`: Number of image observations per timestep (third-person + wrist camera)
- `use_proprio`: Include proprioceptive state information

### Training Configuration
- `adversarial_ratio`: Proportion of adversarial examples in each batch (0.3 = 30%)
- `batch_size`: Training batch size (smaller than standard OpenVLA due to OFT overhead)
- `learning_rate`: Learning rate for optimization
- `num_steps_before_decay`: Steps before learning rate decay begins
- `max_steps`: Total training steps
- `save_freq`: Checkpoint saving frequency

### Fine-tuning Configuration
- `lora_rank`: Rank of LoRA adapter matrices
- `image_aug`: Enable/disable image augmentation
- `save_latest_checkpoint_only`: Whether to keep only the latest checkpoint or save all checkpoints