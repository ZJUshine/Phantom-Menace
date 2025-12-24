# OpenPi Adversarial Training Tutorial

This guide provides step-by-step instructions for conducting adversarial training on the OpenPi (Pi0) model using sensor attack augmentation.

## Prerequisites

### 1. Environment Setup

Install the OpenPi environment following the instructions in the [official OpenPi GitHub repository](https://github.com/Physical-Intelligence/openpi).

### 2. Dataset Preparation

Download the modified LIBERO dataset and place it in your `openpi` directory:

- **Dataset**: [openvla/modified_libero_rlds on Hugging Face](https://huggingface.co/datasets/openvla/modified_libero_rlds)

## Configuration Steps

### 3. Convert Dataset Format

OpenPi uses the LeRobot data format. Convert the LIBERO dataset using the provided conversion script:

**Step 3.1**: Modify the output path in `openpi/examples/libero/convert_libero_data_to_lerobot.py`

**Step 3.2**: Run the conversion script:

```bash
python examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

### 4. Update Training Configuration

Modify the training configuration file to include adversarial training settings:

- **File**: `openpi/src/openpi/training/config.py`
- Reference the modified `config.py` file provided

### 5. Compute Normalization Statistics

Calculate dataset statistics for proper normalization during training:

```bash
python scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune_adversarial
```

This step is essential for ensuring stable training by normalizing observations and actions to appropriate ranges.

### 6. Import Sensor Attack Scripts

Place the sensor attack implementation in the root directory:

- **File Dir**: `sensor_attacks`
- **Destination**: Root directory of OpenPi

### 7. Integrate Sensor Attacks

Modify the data transformation pipeline to include adversarial perturbations:

- **File**: `openpi/src/openpi/transforms.py`
- Reference the modified `transforms.py` file provided

## Training

### 8. Launch Adversarial Training

Run the following command to start adversarial training with the Pi0 model:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_libero_low_mem_finetune_adversarial \
  --exp-name=pi0_libero_low_mem_finetune_adversarial \
  --overwrite
```

## Key Configuration Details

### Environment Variables
- `HF_ENDPOINT`: Hugging Face mirror endpoint for faster downloads (optional)
- `WANDB_BASE_URL`: Weights & Biases API endpoint for experiment tracking
- `XLA_PYTHON_CLIENT_MEM_FRACTION`: Controls memory allocation for JAX/XLA (0.9 = 90% of available GPU memory)

### Training Arguments
- `pi0_libero_low_mem_finetune_adversarial`: Configuration name defined in `config.py`
- `--exp-name`: Experiment name for tracking and checkpointing
- `--overwrite`: Overwrite existing experiment directory if it exists

## Notes

- OpenPi uses JAX/XLA for training, which requires different memory management compared to PyTorch-based models
- The low-memory configuration is optimized for training on GPUs with limited VRAM
- Ensure normalization statistics are computed before training to prevent numerical instability
- The dataset conversion step is necessary because OpenPi's training pipeline expects the LeRobot format