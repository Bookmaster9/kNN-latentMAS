# KV Cache Cosine Similarity Heatmap Visualization

## Overview

This feature generates heatmaps showing cosine similarities between query embeddings and KV cache tokens across all transformer layers during agent transitions in the Latent MAS system.

## What It Does

When enabled with `--show_heatmaps`, the system will:

1. **Compute cosine similarities** between the incoming agent's query embeddings and all cached token keys across **every layer** of the transformer
2. **Generate heatmaps** for three specific agent transitions:
   - **Planner → Critic**
   - **Critic → Refiner**
   - **Refiner → Judger**
3. **Save PNG images** to the `charts/` directory

## Usage

### Basic Command - Multi-Layer Heatmaps

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --max_samples 1 \
    --show_heatmaps
```

### Single-Layer Heatmaps (Middle Layer Only)

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --max_samples 1 \
    --show_heatmaps_singlelayer
```

### Both Multi-Layer and Single-Layer

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --max_samples 1 \
    --show_heatmaps \
    --show_heatmaps_singlelayer
```

### With KNN Filtering

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --max_samples 5 \
    --show_heatmaps_singlelayer \
    --knn_filter \
    --knn_percentage 0.8
```

## Output Files

Heatmaps are saved in the `charts/` directory with the following naming convention:

### Multi-Layer Heatmaps (`--show_heatmaps`)
```
charts/heatmap_planner_to_critic_problem1.png
charts/heatmap_critic_to_refiner_problem1.png
charts/heatmap_refiner_to_judger_problem1.png
charts/heatmap_planner_to_critic_problem2.png
...
```

### Single-Layer Heatmaps (`--show_heatmaps_singlelayer`)
```
charts/heatmap_singlelayer_planner_to_critic_problem1.png
charts/heatmap_singlelayer_critic_to_refiner_problem1.png
charts/heatmap_singlelayer_refiner_to_judger_problem1.png
charts/heatmap_singlelayer_planner_to_critic_problem2.png
...
```

## Understanding the Heatmaps

### Multi-Layer Heatmaps (`--show_heatmaps`)

#### Axes
- **X-axis**: KV Cache Token Position (0 to seq_len)
- **Y-axis**: Transformer Layer Index (0 to num_layers-1)

#### Color Scale
- **Green**: High positive similarity (close to +1)
- **Yellow**: Moderate similarity (close to 0)
- **Red**: Negative similarity (close to -1)

### Single-Layer Heatmaps (`--show_heatmaps_singlelayer`)

#### Axes
- **X-axis**: KV Cache Token Position (0 to seq_len)
- **Y-axis**: Middle Layer (the layer used for kNN filtering)

#### Color Scale (Min-Max Normalized)
- **Dark Red**: Highest similarity (normalized to 1.0)
- **Orange/Yellow**: Medium similarity (normalized to ~0.5)
- **Light Yellow**: Lowest similarity (normalized to 0.0)
- **Title includes original range**: e.g., `Range: [-0.234, 0.891]`

#### Key Differences from Multi-Layer
1. **Single row**: Shows only the middle layer (e.g., layer 16 in a 32-layer model)
2. **Min-max normalized**: Values scaled to [0, 1] for better contrast
3. **Simpler visualization**: Easier to identify which tokens are most relevant
4. **Matches kNN logic**: This is the exact layer used for token selection in kNN filtering

### Interpretation

Each cell `[layer_i, token_j]` shows how similar the incoming agent's query embedding is to token `j` in layer `i` of the KV cache.

**High similarity (green)** indicates:
- The new agent's query is semantically related to that cached token at that layer
- That token is likely relevant for the current agent's reasoning

**Low similarity (red)** indicates:
- The cached token is less relevant to the current agent's query
- Different semantic content or representation

### Layer Patterns

Different layers capture different abstractions:
- **Early layers (0-10)**: Low-level features (syntax, tokens)
- **Middle layers (11-20)**: Mid-level semantics (phrases, local context)
- **Late layers (21-31)**: High-level semantics (abstract reasoning, global context)

You may observe:
- **Horizontal bands**: Certain layers show consistent high/low similarity across all tokens
- **Vertical bands**: Certain token ranges are universally important/unimportant
- **Localized hotspots**: Specific tokens at specific layers are particularly relevant

## Technical Details

### Similarity Computation

For each layer:
1. Extract keys from KV cache: `[batch, num_heads, seq_len, head_dim]`
2. Average across attention heads: `[batch, seq_len, head_dim]`
3. Project query to match head dimension if needed
4. Compute cosine similarity: `cos(query, key) = (query · key) / (||query|| × ||key||)`

### Agent Transitions Tracked

The system tracks these specific transitions based on the default agent sequence:
1. **Planner → Critic**: How the critic views the planner's decomposition
2. **Critic → Refiner**: How the refiner views the critic's evaluation
3. **Refiner → Judger**: How the judger views the refined solution

Note: The judger agent is called "judger" in the code (not "solver").

## Requirements

Additional Python packages required:
- `matplotlib`
- `seaborn`
- `numpy`

Install with:
```bash
pip install matplotlib seaborn numpy
```

## Limitations

- **Batch size**: Currently optimized for `generate_bs=1`. With larger batch sizes, only the first batch element is visualized.
- **Memory**: Computing similarities for all layers requires additional GPU memory (~10-20% overhead).
- **Performance**: Adds ~2-5 seconds per problem for heatmap generation.

## Example Analysis Workflow

1. Run with heatmaps enabled on a small sample:
   ```bash
   python run.py --method latent_mas --model_name Qwen/Qwen3-4B --max_samples 3 --show_heatmaps
   ```

2. Open generated heatmaps in the `charts/` directory

3. Look for patterns:
   - Which tokens does each agent find most relevant?
   - Do certain layers show stronger patterns?
   - How does KNN filtering affect the similarity distribution?

4. Compare across problems to identify consistent patterns

## Troubleshooting

### No heatmaps generated
- Ensure `--show_heatmaps` flag is set
- Check that `--method latent_mas` is specified (not baseline or text_mas)
- Verify `charts/` directory exists (created automatically)

### Memory errors
- Reduce `--max_samples`
- Use a smaller model
- Enable `--knn_filter` to reduce KV cache size

### Import errors
- Install visualization dependencies: `pip install matplotlib seaborn`

## Citation

If you use this visualization feature in your research, please cite the original LatentMAS paper and mention the heatmap visualization extension.
