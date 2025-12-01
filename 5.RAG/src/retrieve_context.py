import os
import numpy as np
import faiss
faiss.omp_set_num_threads(1)
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import math

BASE_DIR = os.path.dirname(__file__)
INDEX_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "embeddings", "faiss_index"))

@lru_cache(maxsize=1)
def _load_model():
    import torch
    try:
        # Force CPU and disable meta device
        torch.set_default_device('cpu')
        
        # Try loading with explicit parameters to avoid meta tensors
        model = SentenceTransformer(
            'all-MiniLM-L6-v2', 
            device='cpu',
            trust_remote_code=True,
            cache_folder=None
        )
        
        # Ensure model is on CPU and has real weights
        model = model.cpu()
        model.eval()
        
        # Test the model
        test_embedding = model.encode(["test"], convert_to_numpy=True)
        if test_embedding is None or len(test_embedding) == 0:
            raise Exception("Model test encoding failed")
            
        return model
        
    except Exception as e:
        print(f"Error loading sentence transformer: {e}")
        
        # Try alternative approach with transformers directly
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch.nn.functional as F
            
            # Load tokenizer and model separately
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            model = AutoModel.from_pretrained(
                'sentence-transformers/all-MiniLM-L6-v2',
                dtype=torch.float32,
                device_map=None
            )
            model = model.cpu()
            model.eval()
            
            # Create a simple wrapper
            class SimpleEmbeddingModel:
                def __init__(self, tokenizer, model):
                    self.tokenizer = tokenizer
                    self.model = model
                    
                def encode(self, sentences, convert_to_numpy=True, show_progress_bar=False):
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    
                    inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    if convert_to_numpy:
                        return embeddings.cpu().numpy()
                    return embeddings
            
            wrapper = SimpleEmbeddingModel(tokenizer, model)
            
            # Test the wrapper
            test_embedding = wrapper.encode(["test"], convert_to_numpy=True)
            if test_embedding is None or len(test_embedding) == 0:
                raise Exception("Wrapper test encoding failed")
                
            return wrapper
            
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
            
            # Last resort: try a different model
            try:
                model = SentenceTransformer(
                    'paraphrase-MiniLM-L6-v2',
                    device='cpu',
                    trust_remote_code=True
                )
                model = model.cpu()
                model.eval()
                return model
            except Exception as e3:
                print(f"All models failed to load: {e3}")
                raise e3

@lru_cache(maxsize=1)
def _load_resources():
    index_path = os.path.join(INDEX_DIR, "cyber_index.faiss")
    texts_path = os.path.join(INDEX_DIR, "texts.npy")
    meta_path = os.path.join(INDEX_DIR, "metadata.npy")
    if not (os.path.exists(index_path) and os.path.exists(texts_path) and os.path.exists(meta_path)):
        return None, None, None
    index = faiss.read_index(index_path)
    texts = np.load(texts_path, allow_pickle=True)
    metadata = np.load(meta_path, allow_pickle=True)
    return index, texts, metadata

def get_relevant_chunks(query, top_k=3):
    try:
        index, texts, metadata = _load_resources()
        if index is None:
            return []
        
        model = _load_model()
        
        # Encode query with error handling
        q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        
        n = index.ntotal
        if n == 0:
            return []
        
        k = min(int(top_k) if top_k else 3, n)
        D, I = index.search(q_emb, k)
        
        return [(texts[i], metadata[i]) for i in I[0] if i >= 0 and i < len(texts)]
        
    except Exception as e:
        print(f"Error in get_relevant_chunks: {e}")
        return []

@lru_cache(maxsize=1)
def _load_mnli():
    try:
        tok = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        clf = AutoModelForSequenceClassification.from_pretrained(
            'facebook/bart-large-mnli',
            dtype=torch.float32,
            device_map=None
        )
        clf.eval()
        return tok, clf
    except Exception as e:
        print(f"Failed to load MNLI model: {e}")
        return None, None

@torch.no_grad()
def _mnli_entailment_scores(query, candidates, batch_size=8):
    tok, clf = _load_mnli()
    if tok is None or clf is None or not candidates:
        return [0.0] * len(candidates)
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        inputs = tok([query] * len(batch), batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = clf(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        entailment = probs[:, 2]  # label order: contradiction, neutral, entailment
        scores.extend(entailment.cpu().tolist())
    return scores

def get_relevant_chunks_mnli(query, top_k=3, initial_k=10):
    try:
        # First retrieve with embeddings
        initial = get_relevant_chunks(query, top_k=min(initial_k, max(top_k*3, top_k)))
        if not initial:
            return []
        texts_only = [t for t, _ in initial]
        scores = _mnli_entailment_scores(query, texts_only)
        ranked = sorted(zip(initial, scores), key=lambda x: x[1], reverse=True)
        top = [pair[0] for pair in ranked[:top_k]]
        return top
    except Exception as e:
        print(f"Error in get_relevant_chunks_mnli: {e}")
        return get_relevant_chunks(query, top_k)

def rerank_by_mnli(query, pairs, top_k=None):
    """Rerank a list of (text, metadata) pairs by MNLI entailment score.
    Returns the reranked list, optionally truncated to top_k.
    """
    try:
        if not pairs:
            return []
        texts_only = [t for t, _ in pairs]
        scores = _mnli_entailment_scores(query, texts_only)
        ranked = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)
        result = [p for (p, s) in ranked]
        if top_k:
            result = result[:top_k]
        return result
    except Exception as e:
        print(f"Error in rerank_by_mnli: {e}")
        return pairs

if __name__ == "__main__":
    print(get_relevant_chunks("What is the vulnerability found in firewall?"))
