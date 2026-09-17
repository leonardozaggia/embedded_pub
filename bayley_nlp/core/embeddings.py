from typing import Iterable, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model_name: str):
    """Load a model by name (sentence-transformers or transformers format).

    Raises a clear error if required packages are missing.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Install sentence-transformers: pip install sentence-transformers"
        ) from e

    # Check if it's a GPT model (contains 'gpt' in name)
    if 'gpt' in model_name.lower() or 'openai' in model_name.lower():
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"Loading GPT model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Handle special cases for OpenAI models
            if 'gpt2' in model_name:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModel.from_pretrained(model_name)
            return {"type": "gpt", "tokenizer": tokenizer, "model": model, "name": model_name}
        except Exception as e:
            print(f"Failed to load GPT model {model_name}: {e}")
            if "MXFP4" in str(e):
                print("This model requires GPU with MXFP4 support. Try a smaller GPT model.")
            print("Falling back to sentence-transformers...")
            return {"type": "sentence-transformers", "model": SentenceTransformer("all-MiniLM-L6-v2")}

    # Regular sentence-transformers model
    try:
        return {"type": "sentence-transformers", "model": SentenceTransformer(model_name)}
    except Exception as e:
        print(f"Failed to load sentence-transformers model {model_name}: {e}")
        print("Falling back to default model...")
        return {"type": "sentence-transformers", "model": SentenceTransformer("all-MiniLM-L6-v2")}


def embed_items(df, text_column: str = "item_text", model=None) -> np.ndarray:
    """Encode item texts to normalized embeddings.

    Args:
        df: DataFrame with a column of texts.
        text_column: Column name containing text.
        model: Optional loaded model; if None, raise.

    Returns:
        np.ndarray of shape (n_items, dim)
    """
    if model is None:
        raise ValueError("Model is required. Call load_model() first.")

    texts: Iterable[str] = df[text_column].tolist()

    # Handle GPT models
    if isinstance(model, dict) and model.get("type") == "gpt":
        return _embed_with_gpt(texts, model["tokenizer"], model["model"])

    # Handle sentence-transformers models
    elif isinstance(model, dict) and model.get("type") == "sentence-transformers":
        st_model = model["model"]
    else:
        # Legacy support for direct SentenceTransformer objects
        st_model = model

    emb = st_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb


def _embed_with_gpt(texts: Iterable[str], tokenizer, model) -> np.ndarray:
    """Generate embeddings from GPT model using mean pooling."""
    import torch
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings = []
    print("Generating GPT embeddings...")

    with torch.no_grad():
        for text in tqdm(texts):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model outputs
            outputs = model(**inputs)

            # Mean pooling over token embeddings
            embeddings_tensor = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embeddings_tensor.cpu().numpy())

    # Stack and normalize
    embeddings_array = np.vstack(embeddings)
    # L2 normalize
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_array = embeddings_array / norms

    return embeddings_array


def compute_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix from normalized embeddings."""
    S = cosine_similarity(embeddings)
    np.fill_diagonal(S, 1.0)
    return S
