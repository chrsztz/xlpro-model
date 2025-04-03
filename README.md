# Piano Fingering Generation

This repository contains implementations of piano fingering generation algorithms including:

1. Deep Learning-based approaches
2. **Model-Based Reinforcement Learning** approach (NEW)

## Model-Based Reinforcement Learning for Piano Fingering

The new implementation is based on the paper:
**"Generating Fingerings for Piano Music with Model-Based Reinforcement Learning"**
by Wanxiang Gao, Sheng Zhang, Nanxi Zhang, Xiaowu Xiong, Zhaojun Shi, and Ka Sun (Applied Sciences, 2023).

### Key Features

- **Environment Model**: A realistic simulation of piano keyboard and hand interaction
- **Model-Based RL**: Prioritized Sweeping algorithm for efficient learning
- **Music Score Segmentation**: For parallel processing of long pieces
- **Elimination of physically impossible fingerings**
- **Optimization for minimal hand movement**

### How it Works

The piano fingering task is modeled as a Markov Decision Process (MDP) where:
- **States**: Position in the score, current fingering, next notes to play
- **Actions**: Fingering choices for the current note/chord
- **Rewards**: Based on fingering difficulty, prioritizing minimal motion
- **Transitions**: Deterministic, moving to the next note/chord

The algorithm uses **invalid action masking** to eliminate physically impossible fingerings
and applies a detailed reward function that quantifies concepts like:
- Finger stretching/contraction rates
- Hand movement distances
- Cross fingering distances
- Fingering mismatches and inversions

### Usage

#### Command Line

```bash
# Use the deep learning approach (default)
python predict_fingering.py path/to/score.musicxml

# Use the reinforcement learning approach
python predict_fingering.py path/to/score.musicxml --use_rl

# Use the RL approach with custom output path
python predict_fingering.py path/to/score.musicxml --use_rl --output custom_output.musicxml
```

#### Test Script

A convenience test script is provided to demonstrate the RL approach on Fur Elise:

```bash
python test_rl_fingering.py
```

### Implementation Structure

The RL-based fingering implementation is in the `rl_fingering_model` folder:

- `rl_fingering_model/utils.py`: Utilities for keyboard and hand simulation
- `rl_fingering_model/environment.py`: MDP environment for piano fingering
- `rl_fingering_model/reinforcement.py`: Prioritized Sweeping algorithm
- `rl_fingering_model/segmentation.py`: Music score segmentation
- `rl_fingering_model/main.py`: Main processing pipeline
- `rl_fingering_model/example.py`: Example usage

### Results

The RL-based approach has several advantages over previous methods:
1. Near elimination of physically impossible fingerings
2. Optimized hand stretching and more efficient motion
3. Increased sampling efficiency through model-based RL

### Requirements

- Python 3.7+
- numpy
- music21
- matplotlib (for visualization)

## Comparison with Other Approaches

The repository now contains multiple approaches to piano fingering generation:

| Approach | Pros | Cons |
|----------|------|------|
| Deep Learning | Can learn from human fingering data, handles complex patterns | Requires large annotated datasets, can generate unplayable fingerings |
| Model-Based RL | Eliminates unplayable fingerings, optimizes for comfort, no training data needed | May not capture stylistic preferences, computationally intensive |

## Citation

If you use this code in your research, please cite the original paper:

```
Gao, W.; Zhang, S.; Zhang, N.; Xiong, X.; Shi, Z.; Sun, K. Generating Fingerings for Piano Music with Model-Based Reinforcement Learning. Appl. Sci. 2023, 13, 11321. https://doi.org/10.3390/app132011321
```
