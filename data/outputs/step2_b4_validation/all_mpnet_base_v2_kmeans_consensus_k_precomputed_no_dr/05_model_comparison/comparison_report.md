# Measurement Model Comparison Report

## Overview

Comparison of data-driven clustering solution to reference measurement models.

## Metrics Summary

| Metric | Theoretical Model | Empirical Model |
|--------|------------------|----------------|
| **Adjusted Rand Index (ARI)** | 0.494 | 0.419 |
| **Normalized Mutual Info (NMI)** | 0.629 | 0.564 |
| **Items Compared** | 42 | 35 |
| **Detected Clusters** | 5 | 5 |
| **Reference Factors** | 5 | 5 |

## Interpretation

**Adjusted Rand Index (ARI)**:
- Range: -1 to 1 (1 = perfect agreement, 0 = random)
- Values > 0.5 indicate strong agreement
- Values > 0.7 indicate very strong agreement

**Normalized Mutual Information (NMI)**:
- Range: 0 to 1 (1 = perfect agreement)
- Values > 0.5 indicate moderate-to-strong shared information

**Conclusion**: The detected clustering shows stronger alignment with the **theoretical model**.
