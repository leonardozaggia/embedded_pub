"""
Model Specification Manager

Utilities for loading, saving, and validating factorial model specifications.
These specifications are outputs from embedding analysis that become inputs for IRT validation.

Usage:
    from bayley_nlp.models import ModelSpecManager
    
    # Load specifications
    manager = ModelSpecManager("config/model_specs.yaml")
    mpnet_model = manager.get_model("mpnet_4factor")
    
    # Save new model from analysis results
    manager.save_model(
        name="mpnet_4factor_v2",
        source="embedding_pca_varimax",
        factor_assignments=primary_factors,
        item_ids=item_ids,
        metadata={"omega": omega_values, "rotation": "varimax"}
    )
    
    # Get items for a specific factor
    items_f1 = manager.get_factor_items("mpnet_4factor", "F1")
    
    # Compare two models
    comparison = manager.compare_models("mpnet_4factor", "roberta_5factor")
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import numpy as np
from datetime import datetime


class ModelSpecManager:
    """Manage factorial model specifications for psychometric validation."""
    
    def __init__(self, config_path: Union[str, Path] = "config/model_specs.yaml"):
        """
        Initialize the model specification manager.
        
        Args:
            config_path: Path to model specifications YAML file
        """
        self.config_path = Path(config_path)
        self.models = self._load_specs()
    
    def _load_specs(self) -> Dict[str, Any]:
        """Load model specifications from YAML file."""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            specs = yaml.safe_load(f) or {}
        
        # Filter out None values and comments
        return {k: v for k, v in specs.items() if v is not None}
    
    def _save_specs(self):
        """Save model specifications to YAML file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.models,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get model specification by name.
        
        Args:
            name: Model name (e.g., "mpnet_4factor")
            
        Returns:
            Model specification dictionary or None if not found
        """
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def get_factor_items(self, model_name: str, factor: str) -> List[str]:
        """
        Get item IDs for a specific factor.
        
        Args:
            model_name: Model name
            factor: Factor name (e.g., "F1", "F2")
            
        Returns:
            List of item IDs (e.g., ["COG_034", "COG_035"])
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        factor_items = model.get("factor_items", {})
        return factor_items.get(factor, [])
    
    def get_factor_indices(self, model_name: str, factor: str) -> List[int]:
        """
        Get item indices for a specific factor.
        
        Args:
            model_name: Model name
            factor: Factor name (e.g., "F1", "F2")
            
        Returns:
            List of item indices (1-based)
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        factor_indices = model.get("factor_indices", {})
        return factor_indices.get(factor, [])
    
    def save_model(
        self,
        name: str,
        source: str,
        factor_assignments: np.ndarray,
        item_ids: List[str],
        model_name: str = "all-mpnet-base-v2",
        rotation: str = "varimax",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save a new model specification from analysis results.
        
        Args:
            name: Unique model name
            source: Source of model (e.g., "embedding_pca_varimax")
            factor_assignments: Array of primary factor assignments (0-indexed)
            item_ids: List of item IDs
            model_name: Embedding model used
            rotation: Rotation method
            metadata: Additional metadata (omega, fit statistics, etc.)
        """
        n_factors = len(np.unique(factor_assignments))
        
        # Build factor mappings
        factor_indices = {}
        factor_items = {}
        
        for f in range(n_factors):
            factor_key = f"F{f+1}"
            items_in_factor = np.where(factor_assignments == f)[0]
            
            # Store 1-based indices (relative to COG_034)
            factor_indices[factor_key] = [int(i + 1) for i in items_in_factor]
            
            # Store item IDs
            factor_items[factor_key] = [item_ids[i] for i in items_in_factor]
        
        # Build specification
        spec = {
            "source": source,
            "model": model_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "n_factors": n_factors,
            "rotation": rotation,
            "factor_indices": factor_indices,
            "factor_items": factor_items,
        }
        
        # Add metadata if provided
        if metadata:
            spec.update(metadata)
        
        # Save to models dict and file
        self.models[name] = spec
        self._save_specs()
        
        print(f"Saved model specification: {name}")
        print(f"  {n_factors} factors, {len(item_ids)} items")
    
    def compare_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """
        Compare two model specifications.
        
        Args:
            model1: First model name
            model2: Second model name
            
        Returns:
            Comparison dictionary with overlap statistics
        """
        m1 = self.get_model(model1)
        m2 = self.get_model(model2)
        
        if m1 is None or m2 is None:
            raise ValueError("Both models must exist")
        
        # Compare number of factors
        comparison = {
            "model1": model1,
            "model2": model2,
            "n_factors": {
                "model1": m1.get("n_factors"),
                "model2": m2.get("n_factors"),
            },
            "factor_overlap": {},
        }
        
        # Calculate overlap between factors
        items1 = m1.get("factor_items", {})
        items2 = m2.get("factor_items", {})
        
        for f1_name, f1_items in items1.items():
            comparison["factor_overlap"][f1_name] = {}
            set1 = set(f1_items)
            
            for f2_name, f2_items in items2.items():
                set2 = set(f2_items)
                overlap = len(set1 & set2)
                jaccard = overlap / len(set1 | set2) if len(set1 | set2) > 0 else 0
                
                comparison["factor_overlap"][f1_name][f2_name] = {
                    "overlap_count": overlap,
                    "jaccard_index": round(jaccard, 3),
                }
        
        return comparison
    
    def export_for_r(self, model_name: str, output_path: str):
        """
        Export model specification as R code for easy loading.
        
        Args:
            model_name: Model to export
            output_path: Path to save R script
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        lines = [
            f"# Model specification: {model_name}",
            f"# Source: {model.get('source')}",
            f"# Date: {model.get('date')}",
            "",
        ]
        
        # Export factor indices
        factor_indices = model.get("factor_indices", {})
        for factor, indices in factor_indices.items():
            indices_str = ", ".join(map(str, indices))
            lines.append(f"{factor}_idx <- c({indices_str})")
        
        lines.append("")
        
        # Export factor items
        factor_items = model.get("factor_items", {})
        for factor, items in factor_items.items():
            items_str = '", "'.join(items)
            lines.append(f'{factor}_items <- c("{items_str}")')
        
        lines.append("")
        lines.append(f"# Convert indices to item IDs: COG_{{033 + index}}")
        lines.append(f'# index_to_item <- function(i) sprintf("COG_%03d", 33 + as.integer(i))')
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"Exported {model_name} to {output_path}")


# Convenience function for quick access
def load_model_spec(model_name: str, config_path: str = "config/model_specs.yaml") -> Dict[str, Any]:
    """
    Quick function to load a single model specification.
    
    Args:
        model_name: Model name to load
        config_path: Path to config file
        
    Returns:
        Model specification dictionary
    """
    manager = ModelSpecManager(config_path)
    return manager.get_model(model_name)
