# VLA Robustness: Sensor Attack Evaluation Framework

This repository provides a comprehensive framework for evaluating the robustness of Vision-Language-Action (VLA) models against various sensor attacks in robotic environments. The framework supports multiple VLA models and implements various camera and microphone attacks to assess model vulnerability.

## Overview

The framework evaluates VLA model robustness by:
- Implementing various sensor attacks (camera and microphone)
- Testing attacks on different VLA models (OpenVLA, OpenVLA-OFT, OpenPI)
- Evaluating performance on LIBERO benchmark tasks
- Providing comprehensive logging and visualization

## Supported Models

- **OpenVLA**: Vision-Language-Action model
- **OpenVLA-OFT**: OpenVLA with Online Fine-Tuning
- **OpenPI**: Policy model with websocket serving

## Sensor Attacks

### Camera Attacks
- **Laser Blinding**: Simulates laser interference on camera sensors
- **Ultrasound Blur**: Applies motion blur effects from ultrasonic interference
- **EM Truncation**: Truncates image data simulating electromagnetic interference
- **Light Projection**: Projects watermarks or patterns onto images
- **Laser Color Strip**: Applies color stripping effects from laser interference
- **EM Color Strip**: Creates color stripe patterns from electromagnetic interference

### Microphone Attacks
- **Voice DOS**: Denial of service attack that empties voice instructions
- **Voice Spoofing**: Injects adversarial instructions into voice commands

## Installation

### Prerequisites

1. **Python Environment**
   ```bash
    conda create -n openvla python=3.10 -y
    conda create -n openvla-oft python=3.10 -y
    conda create -n openpi python=3.11 -y
    conda create -n libero python=3.8 -y
   ```

2. **Install Dependencies**
   ```bash 
   # Install LIBERO simulation environment
   # Follow LIBERO installation instructions at: https://github.com/Lifelong-Robot-Learning/LIBERO
   
   # Install OpenVLA dependencies (if using OpenVLA models)
   # Follow OpenVLA installation instructions
   
   # Install OpenPI dependencies (if using OpenPI models)
   # Follow OpenPI installation instructions
   ```

3. **Setup LIBERO**
   ```bash
   # Clone and install LIBERO
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
   cd LIBERO
   pip install -e .
   ```

## Usage

### 1. OpenVLA Model Evaluation

```bash
python openvla_libero_sensor_attack.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --attack_type laser_blinding \
    --attack_strength weak \
    --center_crop True \
    --use_wandb True \
    --wandb_project vla-robustness-openvla \
    --wandb_entity your_entity
```

### 2. OpenVLA-OFT Model Evaluation

```bash
python openvla-oft_libero_sensor_attack.py \
    --model_family openvla \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --attack_type ultrasound_blur \
    --attack_strength medium \
    --use_l1_regression True \
    --lora_rank 32 \
    --use_wandb True \
    --wandb_project vla-robustness-openvla-oft \
    --wandb_entity your_entity
```

### 3. OpenPI Model Evaluation

First, start the OpenPI policy server:
```bash
python openpi_serve_policy.py \
    --checkpoint.config pi0_libero \
    --checkpoint.path /path/to/checkpoint \
    --env_mode libero_pi0 \
    --host 0.0.0.0 \
    --port 8000
```

Then run the evaluation:
```bash
python openpi_libero_sensro_attack.py \
    --host 0.0.0.0 \
    --port 8000 \
    --task_suite_name libero_spatial \
    --attack_type em_truncation \
    --attack_strength strong \
    --use_wandb True \
    --wandb_project vla-robustness-openpi \
    --wandb_entity your_entity
```

## Configuration Parameters

### Common Parameters

- `--task_suite_name`: LIBERO task suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`)
- `--attack_type`: Type of sensor attack (`none`, `laser_blinding`, `ultrasound_blur`, `em_truncation`, `light_projection`, `laser_color_strip`, `em_color_strip`, `voice_dos`, `voice_spoofing`)
- `--attack_strength`: Attack intensity (`weak`, `medium`, `strong`)
- `--num_trials_per_task`: Number of evaluation episodes per task (default: 10)
- `--seed`: Random seed for reproducibility (default: 7)

### Logging Parameters

- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity name
- `--local_log_dir`: Local directory for log files (default: `./experiments/logs`)

### Attack-Specific Parameters

Attack parameters are automatically configured based on `attack_strength`:

**Laser Blinding:**
- Weak: α = 0.1
- Medium: α = 0.5  
- Strong: α = 0.9

**Ultrasound Blur:**
- Weak: dx=5, dy=5
- Medium: dx=10, dy=10
- Strong: dx=20, dy=20

**EM Truncation:**
- Weak: truncate_ratio = 0.1
- Medium: truncate_ratio = 0.2
- Strong: truncate_ratio = 0.3

## Attack Patterns

The framework includes pre-defined attack patterns in `sensor_attacks/patterns/`:
- `red.png`: Red laser pattern for laser blinding attacks
- `green.png`: Green laser pattern for laser blinding attacks  
- `can.png`: Watermark pattern for light projection attacks

## Output

### Video Recordings
- Rollout videos are saved in `rollouts/` directory
- Videos include attack type and strength in filename
- Success/failure indicated in filename

### Logging
- Local logs saved to `./experiments/logs/`
- W&B logging includes:
  - Per-task success rates
  - Overall success rates
  - Attack parameters
  - Episode counts

### Metrics
- Success rate per task
- Overall success rate across all tasks
- Attack effectiveness measurements

## File Structure

```
vla-robustness/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── openpi_libero_sensro_attack.py        # OpenPI evaluation script
├── openvla_libero_sensor_attack.py       # OpenVLA evaluation script  
├── openvla-oft_libero_sensor_attack.py   # OpenVLA-OFT evaluation script
├── openpi_serve_policy.py                # OpenPI policy server
├── sensor_attacks/                        # Attack implementation modules
│   ├── laser_blinding.py                 # Laser blinding attack
│   ├── ultrasound_blur.py                # Ultrasound blur attack
│   ├── em_truncation.py                  # EM truncation attack
│   ├── light_projection.py               # Light projection attack
│   ├── laser_color_strip.py              # Laser color strip attack
│   ├── em_color_strip.py                 # EM color strip attack
│   └── patterns/                          # Attack pattern images
│       ├── red.png
│       ├── green.png
│       └── can.png
├── rollouts/                              # Video recordings output
├── experiments/                           # Experiment logs and results
└── wandb/                                # W&B logging cache
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- LIBERO benchmark suite for robotic simulation
- OpenVLA and OpenPI model implementations
- Weights & Biases for experiment tracking