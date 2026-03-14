"""
tokenizer/vocab_mapper.py — Mapping entre vocabulaires pour la distillation.
Projette les logits du teacher (LLaMA 128K) vers le student (BPE 32K).
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from tokenizer import BPETokenizer


class SharedVocabMapper:
    """
    Maps between two tokenizer vocabularies for knowledge distillation.
    Supports projecting logits from teacher vocab to student vocab.
    """
    
    def __init__(self, our_tokenizer: BPETokenizer, llama_tokenizer, cache_path: Optional[str] = None):
        """
        Build correspondence matrix between vocabularies.
        
        Args:
            our_tokenizer: Our BPE tokenizer (32K vocab)
            llama_tokenizer: HuggingFace LLaMA tokenizer (128K vocab)
            cache_path: Optional path to save/load mapping cache
        """
        self.our_tok = our_tokenizer
        self.llama_tok = llama_tokenizer
        self.our_vocab_size = our_tokenizer.size
        self.llama_vocab_size = len(llama_tokenizer)
        
        self.our_to_llama: Dict[int, List[int]] = {}
        self.llama_to_our: Dict[int, List[int]] = {}
        self.projection_matrix: Optional[torch.Tensor] = None
        
        if cache_path and Path(cache_path).exists():
            self._load_cache(cache_path)
        else:
            self._build_mapping()
            if cache_path:
                self._save_cache(cache_path)
    
    def _build_mapping(self):
        """Build bidirectional mapping between vocabularies."""
        print(f"[VOCAB] Building mapping: {self.our_vocab_size} ↔ {self.llama_vocab_size} tokens")
        
        self.our_to_llama = defaultdict(list)
        self.llama_to_our = defaultdict(list)
        
        for our_id in range(self.our_vocab_size):
            try:
                if our_id < 256:
                    token_bytes = bytes([our_id])
                elif our_id < self.our_tok.n_base:
                    continue
                else:
                    token_bytes = self.our_tok.vocab.get(our_id, b"")
                
                if not token_bytes:
                    continue
                
                try:
                    token_str = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                
                llama_ids = self.llama_tok.encode(token_str, add_special_tokens=False)
                if llama_ids:
                    self.our_to_llama[our_id] = llama_ids
                    for lid in llama_ids:
                        if our_id not in self.llama_to_our[lid]:
                            self.llama_to_our[lid] = self.llama_to_our.get(lid, []) + [our_id]
            except Exception:
                continue
        
        for llama_id in range(min(self.llama_vocab_size, 50000)):
            if llama_id in self.llama_to_our and self.llama_to_our[llama_id]:
                continue
            try:
                token_str = self.llama_tok.decode([llama_id])
                if not token_str or token_str.startswith("<"):
                    continue
                
                our_ids = self.our_tok.encode(token_str, add_special=False)
                if our_ids:
                    self.llama_to_our[llama_id] = our_ids
            except Exception:
                continue
        
        self.our_to_llama = dict(self.our_to_llama)
        self.llama_to_our = dict(self.llama_to_our)
        
        mapped_our = len([k for k in self.our_to_llama if self.our_to_llama[k]])
        mapped_llama = len([k for k in self.llama_to_our if self.llama_to_our[k]])
        print(f"[VOCAB] Mapped: {mapped_our}/{self.our_vocab_size} our tokens, "
              f"{mapped_llama}/{self.llama_vocab_size} LLaMA tokens")
    
    def _save_cache(self, path: str):
        """Save mapping to cache file."""
        import json
        data = {
            "our_vocab_size": self.our_vocab_size,
            "llama_vocab_size": self.llama_vocab_size,
            "our_to_llama": {str(k): v for k, v in self.our_to_llama.items()},
            "llama_to_our": {str(k): v for k, v in self.llama_to_our.items()}
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[VOCAB] Cache saved → {path}")
    
    def _load_cache(self, path: str):
        """Load mapping from cache file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self.our_to_llama = {int(k): v for k, v in data["our_to_llama"].items()}
        self.llama_to_our = {int(k): v for k, v in data["llama_to_our"].items()}
        print(f"[VOCAB] Cache loaded ← {path}")
    
    def build_projection_matrix(self, device="cpu") -> torch.Tensor:
        """
        Build sparse projection matrix M where M[our_id, llama_id] = weight.
        Shape: (our_vocab_size, llama_vocab_size)
        """
        if self.projection_matrix is not None:
            return self.projection_matrix.to(device)
        
        print("[VOCAB] Building projection matrix...")
        
        rows, cols, vals = [], [], []
        
        for llama_id, our_ids in self.llama_to_our.items():
            if not our_ids:
                continue
            weight = 1.0 / len(our_ids)
            for our_id in our_ids:
                rows.append(our_id)
                cols.append(llama_id)
                vals.append(weight)
        
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        
        self.projection_matrix = torch.sparse_coo_tensor(
            indices, values,
            size=(self.our_vocab_size, self.llama_vocab_size)
        ).coalesce()
        
        print(f"[VOCAB] Projection matrix: {len(vals)} non-zero entries")
        return self.projection_matrix.to(device)
    
    def project_teacher_logits(self, llama_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Project teacher logits from LLaMA vocab (128K) to our vocab (32K).
        
        Args:
            llama_logits: Tensor of shape (B, T, llama_vocab_size) or (B, llama_vocab_size)
            temperature: Temperature for softmax
        
        Returns:
            Tensor of shape (B, T, our_vocab_size) or (B, our_vocab_size)
        """
        device = llama_logits.device
        original_shape = llama_logits.shape
        
        if len(original_shape) == 3:
            B, T, V = original_shape
            llama_logits = llama_logits.view(B * T, V)
        else:
            B, V = original_shape
            T = None
        
        llama_probs = F.softmax(llama_logits / temperature, dim=-1)
        
        M = self.build_projection_matrix(device)
        
        if M.is_sparse:
            M_dense = M.to_dense()
        else:
            M_dense = M
        
        our_probs = torch.mm(llama_probs, M_dense.t())
        
        our_probs = our_probs / (our_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        our_logits = torch.log(our_probs + 1e-10) * temperature
        
        if T is not None:
            our_logits = our_logits.view(B, T, self.our_vocab_size)
        
        return our_logits
    
    def project_teacher_probs(self, llama_probs: torch.Tensor) -> torch.Tensor:
        """
        Project teacher probabilities from LLaMA vocab to our vocab.
        
        Args:
            llama_probs: Tensor of shape (..., llama_vocab_size)
        
        Returns:
            Tensor of shape (..., our_vocab_size)
        """
        device = llama_probs.device
        original_shape = llama_probs.shape
        
        flat_probs = llama_probs.view(-1, original_shape[-1])
        
        M = self.build_projection_matrix(device)
        if M.is_sparse:
            M_dense = M.to_dense()
        else:
            M_dense = M
        
        our_probs = torch.mm(flat_probs, M_dense.t())
        our_probs = our_probs / (our_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        new_shape = list(original_shape[:-1]) + [self.our_vocab_size]
        return our_probs.view(*new_shape)


class TextAlignedMapper:
    """
    Alternative mapper using text-based alignment.
    For each text, tokenize with both tokenizers and align by character position.
    """
    
    def __init__(self, our_tokenizer: BPETokenizer, llama_tokenizer):
        self.our_tok = our_tokenizer
        self.llama_tok = llama_tokenizer
    
    def align_tokens(self, text: str) -> List[Tuple[int, List[int]]]:
        """
        Align our tokens with LLaMA tokens by character position.
        
        Returns:
            List of (our_token_id, [llama_token_ids]) pairs
        """
        our_ids = self.our_tok.encode(text, add_special=False)
        llama_encoding = self.llama_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        llama_ids = llama_encoding["input_ids"]
        llama_offsets = llama_encoding["offset_mapping"]
        
        our_offsets = []
        pos = 0
        for our_id in our_ids:
            if our_id < 256:
                token_len = 1
            elif our_id < self.our_tok.n_base:
                token_len = 0
            else:
                token_bytes = self.our_tok.vocab.get(our_id, b"")
                try:
                    token_len = len(token_bytes.decode("utf-8"))
                except:
                    token_len = len(token_bytes)
            our_offsets.append((pos, pos + token_len))
            pos += token_len
        
        alignments = []
        for our_idx, (our_start, our_end) in enumerate(our_offsets):
            matched_llama = []
            for llama_idx, (l_start, l_end) in enumerate(llama_offsets):
                if l_end > our_start and l_start < our_end:
                    matched_llama.append(llama_ids[llama_idx])
            alignments.append((our_ids[our_idx], matched_llama))
        
        return alignments
    
    def create_position_mapping(self, text: str) -> torch.Tensor:
        """
        Create a mapping matrix for a specific text.
        
        Returns:
            Sparse tensor of shape (our_seq_len, llama_seq_len)
        """
        alignments = self.align_tokens(text)
        
        our_len = len(alignments)
        llama_ids = self.llama_tok.encode(text, add_special_tokens=False)
        llama_len = len(llama_ids)
        
        llama_id_to_pos = defaultdict(list)
        for pos, lid in enumerate(llama_ids):
            llama_id_to_pos[lid].append(pos)
        
        rows, cols, vals = [], [], []
        
        for our_pos, (our_id, matched_llama_ids) in enumerate(alignments):
            if not matched_llama_ids:
                continue
            weight = 1.0 / len(matched_llama_ids)
            for lid in matched_llama_ids:
                for llama_pos in llama_id_to_pos.get(lid, []):
                    rows.append(our_pos)
                    cols.append(llama_pos)
                    vals.append(weight)
        
        if not rows:
            return torch.zeros(our_len, llama_len)
        
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        
        return torch.sparse_coo_tensor(
            indices, values, size=(our_len, llama_len)
        ).coalesce()


def create_vocab_mapper(our_tokenizer_path: str, llama_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Factory function to create a SharedVocabMapper.
    
    Args:
        our_tokenizer_path: Path to our tokenizer directory
        llama_model_name: HuggingFace model name for LLaMA tokenizer
    
    Returns:
        SharedVocabMapper instance
    """
    from transformers import AutoTokenizer
    
    our_tok = BPETokenizer.load(our_tokenizer_path)
    llama_tok = AutoTokenizer.from_pretrained(llama_model_name)
    
    cache_path = Path(our_tokenizer_path) / "vocab_mapping_cache.json"
    
    return SharedVocabMapper(our_tok, llama_tok, str(cache_path))


if __name__ == "__main__":
    print("Vocab Mapper Test")
    print("="*50)
    
    from config import TOKENIZER_DIR
    
    try:
        our_tok = BPETokenizer.load(str(TOKENIZER_DIR))
        print(f"Our tokenizer: {our_tok.size} tokens")
        
        try:
            from transformers import AutoTokenizer
            llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            print(f"LLaMA tokenizer: {len(llama_tok)} tokens")
            
            mapper = SharedVocabMapper(our_tok, llama_tok)
            
            import torch
            fake_llama_logits = torch.randn(2, 10, len(llama_tok))
            our_logits = mapper.project_teacher_logits(fake_llama_logits)
            print(f"\nProjection test:")
            print(f"  Input:  {fake_llama_logits.shape}")
            print(f"  Output: {our_logits.shape}")
            print("✓ Projection OK")
            
        except ImportError:
            print("transformers not installed - skipping LLaMA tokenizer test")
        except Exception as e:
            print(f"Could not load LLaMA tokenizer: {e}")
            
    except FileNotFoundError:
        print(f"Our tokenizer not found at {TOKENIZER_DIR}")
        print("Train the tokenizer first with: python -m tokenizer.train_tokenizer")
