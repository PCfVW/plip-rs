#!/usr/bin/env python3
"""RWKV-6 reference validation: standalone forward pass + generation.

Loads weights from safetensors and runs the RWKV-6 algorithm directly
(no dependency on the buggy HF custom modeling code).

Usage:
    python scripts/rwkv6_validation.py

Output:
    scripts/rwkv6_reference.json — reference data for Rust validation
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file


MODEL_ID = "RWKV/v6-Finch-1B6-HF"
TEST_PROMPT = "def fibonacci(n):\n    "

# Config for v6-Finch-1B6-HF
HIDDEN_SIZE = 2048
NUM_LAYERS = 24
HEAD_SIZE = 64
NUM_HEADS = HIDDEN_SIZE // HEAD_SIZE  # 32
VOCAB_SIZE = 65536
INTERMEDIATE_SIZE = (HIDDEN_SIZE * 7 // 2) // 32 * 32  # 7168
LAYER_NORM_EPS = 1e-5
HEAD_SIZE_DIVISOR = 8
TIME_MIX_EXTRA_DIM = 32
TIME_DECAY_EXTRA_DIM = 64


def ensure_safetensors():
    """Ensure model.safetensors exists, converting from pytorch_model.bin if needed."""
    print("=" * 60)
    print("Ensuring safetensors weights exist")
    print("=" * 60)

    bin_path = Path(hf_hub_download(MODEL_ID, "pytorch_model.bin"))
    out_path = bin_path.parent / "model.safetensors"

    if out_path.exists():
        print(f"model.safetensors already exists at {out_path}")
        return str(out_path)

    print(f"Converting {bin_path} -> {out_path}...")
    weights = torch.load(bin_path, map_location="cpu", weights_only=True)
    save_file(weights, str(out_path))
    print(f"Done. Size: {out_path.stat().st_size / 1e9:.2f} GB")
    return str(out_path)


def load_tokenizer():
    """Load the RWKV tokenizer from vocab file (same algorithm as our Rust code)."""
    vocab_path = hf_hub_download(MODEL_ID, "rwkv_vocab_v20230424.txt")
    idx2token = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_space = line.index(" ")
            idx = int(line[:first_space])
            rest = line[first_space + 1:]
            last_space = rest.rindex(" ")
            token_repr = rest[:last_space]
            # Evaluate the Python literal
            token_bytes = eval(token_repr)  # noqa: S307
            if isinstance(token_bytes, str):
                token_bytes = token_bytes.encode("utf-8")
            idx2token[idx] = token_bytes
    return idx2token


def encode(text, idx2token):
    """Greedy longest-match encoding (same algorithm as Rust Trie)."""
    # Build reverse map for quick lookup
    token2idx = {}
    for idx, token_bytes in idx2token.items():
        token2idx[token_bytes] = idx

    src = text.encode("utf-8")
    tokens = []
    i = 0
    while i < len(src):
        best_len = 0
        best_id = None
        # Try all lengths from longest to shortest
        for end in range(min(i + 128, len(src)), i, -1):
            candidate = bytes(src[i:end])
            if candidate in token2idx:
                best_len = end - i
                best_id = token2idx[candidate]
                break
        if best_id is None:
            raise ValueError(f"No matching token at position {i}")
        tokens.append(best_id)
        i += best_len
    return tokens


def decode_tokens(ids, idx2token):
    """Decode token IDs back to string."""
    result = b""
    for tid in ids:
        result += idx2token[tid]
    return result.decode("utf-8", errors="replace")


def layer_norm(x, weight, bias, eps=LAYER_NORM_EPS):
    """Standard LayerNorm."""
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


def group_norm(x, weight, bias, num_heads, head_size):
    """GroupNorm matching RWKV's ln_x (per-head normalization).

    nn.GroupNorm(num_heads, hidden_size, eps) expects input (N, C, *).
    """
    B, T, C = x.shape
    eps = LAYER_NORM_EPS * HEAD_SIZE_DIVISOR * HEAD_SIZE_DIVISOR
    x = x.view(B * T, C).unsqueeze(-1)  # [B*T, C, 1]
    x = F.group_norm(x, num_groups=num_heads, weight=weight, bias=bias, eps=eps)
    return x.squeeze(-1).view(B, T, C)


def rwkv6_attention(hidden, state_attn_x, state_attn_kv, layer_weights, layer_idx):
    """RWKV-6 time-mix (attention) forward pass."""
    B, T, C = hidden.shape
    H = NUM_HEADS
    S = HEAD_SIZE

    w = layer_weights

    # Token shift
    if state_attn_x is not None:
        shifted = torch.cat([state_attn_x.unsqueeze(1), hidden[:, :-1, :]], dim=1)
    else:
        shifted = torch.cat([torch.zeros(B, 1, C, dtype=hidden.dtype), hidden[:, :-1, :]], dim=1)

    # Save last token as new state
    new_attn_x = hidden[:, -1, :].clone()

    xx = shifted - hidden

    # Data-dependent mixing
    xxx = hidden + xx * w["time_maa_x"]
    xxx = torch.tanh(xxx @ w["time_maa_w1"])  # [B, T, 160]
    xxx = xxx.view(B * T, 5, -1).transpose(0, 1)  # [5, B*T, 32]
    xxx = torch.bmm(xxx, w["time_maa_w2"])  # [5, B*T, C]
    xxx = xxx.view(5, B, T, -1)  # [5, B, T, C]

    mw, mk, mv, mr, mg = xxx.unbind(dim=0)  # each [B,T,C]

    # Mix inputs
    xw = hidden + xx * (w["time_maa_w"] + mw)
    xk = hidden + xx * (w["time_maa_k"] + mk)
    xv = hidden + xx * (w["time_maa_v"] + mv)
    xr = hidden + xx * (w["time_maa_r"] + mr)
    xg = hidden + xx * (w["time_maa_g"] + mg)

    # Project R, K, V, gate
    r = (xr @ w["receptance.weight"].t()).view(B, T, H, S)
    k = (xk @ w["key.weight"].t()).view(B, T, H, S)
    v = (xv @ w["value.weight"].t()).view(B, T, H, S)
    g = F.silu(xg @ w["gate.weight"].t())

    # Data-dependent decay
    td = w["time_decay"] + torch.tanh(xw @ w["time_decay_w1"]) @ w["time_decay_w2"]
    td = td.view(B, T, H, S)

    time_faaaa = w["time_faaaa"].view(1, 1, H, S)

    # WKV recurrence
    if state_attn_kv is not None:
        state = state_attn_kv.clone().float()
    else:
        state = torch.zeros(B, H, S, S, dtype=torch.float32)

    out_list = []
    for t in range(T):
        rt = r[:, t, :, :]  # [B, H, S]
        kt = k[:, t, :, :]  # [B, H, S]
        vt = v[:, t, :, :]  # [B, H, S]
        dt = td[:, t, :, :]  # [B, H, S]

        decay = torch.exp(-torch.exp(dt)).float()  # [B, H, S]

        kv = kt.float().unsqueeze(-1) * vt.float().unsqueeze(-2)  # [B,H,S,S]

        time_first_kv = time_faaaa[:, 0, :, :].float().unsqueeze(-1) * kv  # [B,H,S,S]

        ot = (rt.float().unsqueeze(-2) @ (time_first_kv + state)).squeeze(-2)  # [B,H,S]
        state = kv + decay.unsqueeze(-1) * state

        out_list.append(ot.to(hidden.dtype))

    new_attn_kv = state.to(hidden.dtype)

    out = torch.stack(out_list, dim=1)  # [B, T, H, S]
    out = out.view(B, T, C)

    # GroupNorm
    out = group_norm(out, w["ln_x.weight"], w["ln_x.bias"], H, S)

    # Gate and output
    out = out * g
    out = out @ w["output.weight"].t()

    return out, new_attn_x, new_attn_kv


def rwkv6_ffn(hidden, state_ffn_x, layer_weights):
    """RWKV-6 channel-mix (feed-forward) forward pass."""
    B, T, C = hidden.shape
    w = layer_weights

    # Token shift
    if state_ffn_x is not None:
        shifted = torch.cat([state_ffn_x.unsqueeze(1), hidden[:, :-1, :]], dim=1)
    else:
        shifted = torch.cat([torch.zeros(B, 1, C, dtype=hidden.dtype), hidden[:, :-1, :]], dim=1)

    new_ffn_x = hidden[:, -1, :].clone()

    xx = shifted - hidden
    xk = hidden + xx * w["ffn_time_maa_k"]
    xr = hidden + xx * w["ffn_time_maa_r"]

    k = F.relu(xk @ w["ffn_key.weight"].t()) ** 2
    v = k @ w["ffn_value.weight"].t()
    r = torch.sigmoid(xr @ w["ffn_receptance.weight"].t())

    return r * v, new_ffn_x


def rwkv6_forward(input_ids, weights, idx2token):
    """Full RWKV-6 forward pass returning logits for all positions."""
    B = 1
    T = len(input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # Embed
    hidden = weights["rwkv.embeddings.weight"][input_tensor]  # [1, T, C]
    hidden = hidden.float()  # Work in float32 for numerical stability

    # Initialize state (all zeros)
    states_attn_x = [None] * NUM_LAYERS
    states_attn_kv = [None] * NUM_LAYERS
    states_ffn_x = [None] * NUM_LAYERS

    for layer_idx in range(NUM_LAYERS):
        prefix = f"rwkv.blocks.{layer_idx}"

        # Pre-LN for first block
        if layer_idx == 0:
            hidden = layer_norm(
                hidden,
                weights[f"{prefix}.pre_ln.weight"].float(),
                weights[f"{prefix}.pre_ln.bias"].float(),
            )

        # Collect attention weights
        attn_w = {}
        for key in ["time_maa_x", "time_maa_w", "time_maa_k", "time_maa_v",
                     "time_maa_r", "time_maa_g", "time_maa_w1", "time_maa_w2",
                     "time_decay", "time_decay_w1", "time_decay_w2", "time_faaaa",
                     "receptance.weight", "key.weight", "value.weight",
                     "gate.weight", "output.weight", "ln_x.weight", "ln_x.bias"]:
            attn_w[key] = weights[f"{prefix}.attention.{key}"].float()

        # LN1 + Attention
        ln1_out = layer_norm(
            hidden,
            weights[f"{prefix}.ln1.weight"].float(),
            weights[f"{prefix}.ln1.bias"].float(),
        )
        attn_out, states_attn_x[layer_idx], states_attn_kv[layer_idx] = rwkv6_attention(
            ln1_out, states_attn_x[layer_idx], states_attn_kv[layer_idx], attn_w, layer_idx
        )
        hidden = hidden + attn_out

        # Collect FFN weights
        ffn_w = {
            "ffn_time_maa_k": weights[f"{prefix}.feed_forward.time_maa_k"].float(),
            "ffn_time_maa_r": weights[f"{prefix}.feed_forward.time_maa_r"].float(),
            "ffn_key.weight": weights[f"{prefix}.feed_forward.key.weight"].float(),
            "ffn_value.weight": weights[f"{prefix}.feed_forward.value.weight"].float(),
            "ffn_receptance.weight": weights[f"{prefix}.feed_forward.receptance.weight"].float(),
        }

        # LN2 + FFN
        ln2_out = layer_norm(
            hidden,
            weights[f"{prefix}.ln2.weight"].float(),
            weights[f"{prefix}.ln2.bias"].float(),
        )
        ffn_out, states_ffn_x[layer_idx] = rwkv6_ffn(ln2_out, states_ffn_x[layer_idx], ffn_w)
        hidden = hidden + ffn_out

    # Final LN + head
    hidden = layer_norm(
        hidden,
        weights["rwkv.ln_out.weight"].float(),
        weights["rwkv.ln_out.bias"].float(),
    )
    logits = hidden @ weights["head.weight"].float().t()  # [1, T, vocab]

    return logits, (states_attn_x, states_attn_kv, states_ffn_x)


def rwkv6_generate_greedy(prompt_ids, weights, idx2token, max_new_tokens=20):
    """Generate tokens greedily using the full-sequence forward pass."""
    # Run prompt through the model
    logits, state = rwkv6_forward(prompt_ids, weights, idx2token)

    generated = []
    all_ids = list(prompt_ids)

    for step in range(max_new_tokens):
        # Get next token (greedy = argmax of last position)
        next_logits = logits[0, -1, :]
        next_token = next_logits.argmax().item()

        if next_token == 0:  # EOS
            break

        generated.append(next_token)
        all_ids.append(next_token)

        # Run single-token forward for next step
        # For simplicity, re-run the full sequence (slow but correct for validation)
        logits, state = rwkv6_forward(all_ids, weights, idx2token)

    return generated


def main():
    # Ensure safetensors exist
    safetensors_path = ensure_safetensors()

    print("\n" + "=" * 60)
    print("Loading weights from safetensors")
    print("=" * 60)
    weights = load_file(safetensors_path)
    print(f"Loaded {len(weights)} tensors")

    # Load tokenizer
    print("\nLoading RWKV tokenizer...")
    idx2token = load_tokenizer()
    print(f"Vocabulary size: {len(idx2token)}")

    # Tokenize
    print(f"\nTest prompt: {repr(TEST_PROMPT)}")
    token_ids = encode(TEST_PROMPT, idx2token)
    print(f"Token IDs ({len(token_ids)}): {token_ids}")
    token_strings = [decode_tokens([tid], idx2token) for tid in token_ids]
    print(f"Tokens: {token_strings}")

    # Forward pass
    print("\nRunning forward pass (float32, CPU)...")
    print("(This will take a minute — 24 layers, no GPU)")
    logits, state = rwkv6_forward(token_ids, weights, idx2token)
    print(f"Logits shape: {logits.shape}")

    # Top-10 predictions for next token
    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, 10)

    print(f"\nTop-10 predictions for next token:")
    top_predictions = []
    for i in range(10):
        tid = top_ids[i].item()
        prob = top_probs[i].item()
        token_str = decode_tokens([tid], idx2token)
        logit_val = last_logits[tid].item()
        print(f"  {i+1}. {repr(token_str):>15} (id={tid:>5}, prob={prob:.4f}, logit={logit_val:.4f})")
        top_predictions.append({
            "token_id": tid,
            "token": token_str,
            "probability": prob,
            "logit": logit_val,
        })

    top_logit_values = [last_logits[top_ids[i]].item() for i in range(10)]

    # Greedy generation
    print("\nGenerating 20 tokens (greedy)...")
    print("(This re-runs full forward per token — slow but correct for validation)")
    generated_ids = rwkv6_generate_greedy(token_ids, weights, idx2token, max_new_tokens=20)
    generated_text = decode_tokens(generated_ids, idx2token)
    full_text = decode_tokens(token_ids + generated_ids, idx2token)

    print(f"Generated IDs: {generated_ids}")
    print(f"Generated text: {repr(generated_text)}")
    print(f"Full output: {repr(full_text)}")

    # Save reference
    reference = {
        "model_id": MODEL_ID,
        "test_prompt": TEST_PROMPT,
        "token_ids": token_ids,
        "token_strings": token_strings,
        "top_predictions": top_predictions,
        "top_logit_values": top_logit_values,
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "full_generated_text": full_text,
    }

    out_file = Path(__file__).parent / "rwkv6_reference.json"
    with open(out_file, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nReference data saved to {out_file}")

    print("\n" + "=" * 60)
    print("Validation data ready for Rust comparison")
    print("=" * 60)


if __name__ == "__main__":
    main()
