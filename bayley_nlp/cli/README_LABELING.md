# Cluster Centroid Labeling

## Overview

This module provides tools to generate interpretable labels for clusters by **decoding cluster centroids** back into semantic meaning. The centroid of a cluster represents the average embedding vector of all items in that cluster, capturing the shared semantic content.

## How It Works

### 1. Centroid Computation
For each cluster, we compute the centroid (mean) of all item embeddings in that cluster:
```
centroid_k = mean(embeddings[labels == k])
```

### 2. Semantic Decoding
The centroid is then "decoded" by finding:
- **Nearest items**: The items whose embeddings are most similar to the centroid
- **Cognitive concepts**: Pre-defined domain concepts most similar to the centroid

This reveals what the cluster "means" semantically.

## Usage

### Basic Usage

```bash
python bayley_nlp/cli/04_labeling.py \
  --embeddings data/processed/embeddings_all_mpnet_base_v2.npy \
  --labels-file data/outputs/experiments/all_mpnet_base_v2_analysis/03_clustering/cluster_assignments.csv \
  --items data/processed/items.csv \
  --top-k 7
```

**Note**: The output directory is automatically detected from the model name and follows the pipeline structure:
`data/outputs/experiments/<model_name>_analysis/04_labeling/`

You can override this with `--outdir` if needed.

### With Concept Decoding

To also decode centroids against cognitive domain concepts:

```bash
python bayley_nlp/cli/04_labeling.py \
  --embeddings data/processed/embeddings_all_mpnet_base_v2.npy \
  --labels-file data/outputs/experiments/all_mpnet_base_v2_analysis/03_clustering/cluster_assignments.csv \
  --items data/processed/items.csv \
  --top-k 7 \
  --decode-concepts \
  --visualize
```

## Outputs

The script generates several files in the output directory:

### 1. `cluster_labels_summary.csv`
Quick overview of each cluster:
- `cluster_id`: Cluster number
- `n_items`: Number of items in cluster
- `representative_items`: Top 3 most representative items
- `top_similarity`: Highest similarity to centroid
- `mean_similarity`: Average similarity to centroid

### 2. `cluster_labels_report.md`
Detailed markdown report with:
- Full list of nearest items for each cluster
- Item texts (truncated for readability)
- Similarity scores

### 3. `cluster_concept_labels.csv` (if --decode-concepts used)
Cognitive domain interpretations:
- `cluster_id`: Cluster number
- `top_concept`: Most similar cognitive domain concept
- `top_concept_similarity`: Similarity score
- `all_concepts`: Top 5 concepts with scores

### 4. `cluster_N_nearest_items.csv`
Individual files for each cluster with full item details:
- `item_id`: Item identifier
- `item_text`: Full item description
- `similarity`: Cosine similarity to centroid

## Example Output

For a run with 2 clusters:

**Cluster 0** (Centroid decoded to "sequential problem solving"):
- Top items: Shape board tasks, pegboard tasks
- Interpretation: Tasks requiring spatial reasoning and sequential manipulation

**Cluster 1** (Centroid decoded to "symbolic and pretend play"):
- Top items: Object permanence, hidden object tasks, tool use
- Interpretation: Tasks requiring means-end reasoning and symbolic thinking

## Cognitive Domain Concepts

The script can decode centroids against these pre-defined concepts:
- Object permanence and search
- Spatial reasoning and problem solving
- Fine motor manipulation
- Symbolic and pretend play
- Means-end reasoning and tool use
- Shape and pattern recognition
- Sequential problem solving
- Visual-motor coordination
- Exploration and curiosity
- Early cognitive skills
- Executive function and planning
- Memory and recall
- Cause and effect understanding
- Categorization and sorting
- Imitation and learning

## Interpretation Guide

### High Similarity Scores (> 0.85)
Items are very representative of the cluster's semantic meaning. The cluster is cohesive.

### Medium Similarity Scores (0.70 - 0.85)
Items share semantic features but with some variation. Normal for developmental assessments.

### Low Similarity Scores (< 0.70)
The cluster may be heterogeneous or the centroid may not well-represent individual items.

## Technical Details

### Why Centroids?
Centroids provide a natural summary statistic in embedding space. By averaging the embeddings of items in a cluster and then finding what's most similar to that average, we can identify the "prototypical" items that best represent the cluster's meaning.

### Normalization
All embeddings and centroids are L2-normalized before computing similarities, ensuring that cosine similarity is equivalent to Euclidean distance in the normalized space.

### Relationship to Factor Labels
These centroid-based labels complement traditional factor analysis interpretations:
- **Traditional FA**: Expert examines factor loadings
- **Centroid decoding**: Algorithmic identification of representative items
- **Concept matching**: Maps to pre-defined cognitive domains

Both approaches are useful for interpreting latent dimensions!

## See Also

- `bayley_nlp/pipelines/step2_b4_validation.py`, `step3_b3_bottomup.py`: the pipelines that run this step
- `bayley_nlp/core/clustering.py`: clustering implementation
- `bayley_nlp/core/embeddings.py`: Embedding generation
