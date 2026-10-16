"""Convert an OpenAI Whisper checkpoint (Hugging Face layout) into a tpt-voice
model directory.

    python tools/convert_whisper.py <hf_dir> <out_dir>

``<hf_dir>`` holds the files from e.g. ``openai/whisper-tiny.en``: model.safetensors,
config.json, vocab.json, merges.txt and added_tokens.json. The output directory
gets ``model.safetensors``, ``config.json`` and ``tokenizer.json`` in the layout
``tpt_av_voice_transcribe::weights`` loads.

The conversion is exact apart from two inputs that Whisper does not store:
the key projections have no bias (zeros are written), and the output
projection is tied to the token embedding (as in Whisper itself).
"""

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def load_config(hf: Path) -> dict:
    cfg = json.loads((hf / "config.json").read_text(encoding="utf-8"))
    return {
        "n_mels": cfg["num_mel_bins"],
        "n_audio_ctx": cfg["max_source_positions"],
        "n_audio_state": cfg["d_model"],
        "n_audio_head": cfg["encoder_attention_heads"],
        "n_audio_layer": cfg["encoder_layers"],
        "n_vocab": cfg["vocab_size"],
        "n_text_ctx": cfg["max_target_positions"],
        "n_text_state": cfg["d_model"],
        "n_text_head": cfg["decoder_attention_heads"],
        "n_text_layer": cfg["decoder_layers"],
    }


def load_tokenizer(hf: Path, n_vocab: int) -> dict:
    vocab = json.loads((hf / "vocab.json").read_text(encoding="utf-8"))
    added = json.loads((hf / "added_tokens.json").read_text(encoding="utf-8"))
    by_id = [None] * n_vocab
    for token, idx in {**vocab, **added}.items():
        by_id[idx] = token
    missing = [i for i, t in enumerate(by_id) if t is None]
    if missing:
        raise SystemExit(f"vocabulary has {len(missing)} ids with no token (e.g. {missing[:3]})")

    merges = []
    for line in (hf / "merges.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        merges.append(line)

    specials = [t for t in added if t.startswith("<|") and t.endswith("|>")]
    return {"vocab": by_id, "merges": merges, "specials": specials}


def convert_weights(hf_tensors: dict, cfg: dict) -> dict:
    t = {}

    def get(name):
        return np.asarray(hf_tensors[name], dtype=np.float32)

    zeros = lambda n: np.zeros((n,), dtype=np.float32)

    # Encoder front end.
    t["encoder.conv1.weight"] = get("model.encoder.conv1.weight")
    t["encoder.conv1.bias"] = get("model.encoder.conv1.bias")
    t["encoder.conv2.weight"] = get("model.encoder.conv2.weight")
    t["encoder.conv2.bias"] = get("model.encoder.conv2.bias")
    t["encoder.positional_embedding"] = get("model.encoder.embed_positions.weight")

    d = cfg["n_audio_state"]
    for i in range(cfg["n_audio_layer"]):
        src = f"model.encoder.layers.{i}"
        dst = f"encoder.blocks.{i}"
        _attention(t, get, src + ".self_attn", dst + ".attn", d, zeros)
        t[f"{dst}.attn_ln.weight"] = get(f"{src}.self_attn_layer_norm.weight")
        t[f"{dst}.attn_ln.bias"] = get(f"{src}.self_attn_layer_norm.bias")
        _mlp(t, get, src, dst)
        t[f"{dst}.mlp_ln.weight"] = get(f"{src}.final_layer_norm.weight")
        t[f"{dst}.mlp_ln.bias"] = get(f"{src}.final_layer_norm.bias")

    t["encoder.ln_post.weight"] = get("model.encoder.layer_norm.weight")
    t["encoder.ln_post.bias"] = get("model.encoder.layer_norm.bias")

    # Decoder.
    t["decoder.token_embedding.weight"] = get("model.decoder.embed_tokens.weight")
    t["decoder.positional_embedding"] = get("model.decoder.embed_positions.weight")

    d = cfg["n_text_state"]
    for i in range(cfg["n_text_layer"]):
        src = f"model.decoder.layers.{i}"
        dst = f"decoder.blocks.{i}"
        _attention(t, get, src + ".self_attn", dst + ".attn", d, zeros)
        t[f"{dst}.attn_ln.weight"] = get(f"{src}.self_attn_layer_norm.weight")
        t[f"{dst}.attn_ln.bias"] = get(f"{src}.self_attn_layer_norm.bias")
        _attention(t, get, src + ".encoder_attn", dst + ".cross_attn", d, zeros)
        t[f"{dst}.cross_attn_ln.weight"] = get(f"{src}.encoder_attn_layer_norm.weight")
        t[f"{dst}.cross_attn_ln.bias"] = get(f"{src}.encoder_attn_layer_norm.bias")
        _mlp(t, get, src, dst)
        t[f"{dst}.mlp_ln.weight"] = get(f"{src}.final_layer_norm.weight")
        t[f"{dst}.mlp_ln.bias"] = get(f"{src}.final_layer_norm.bias")

    t["decoder.ln.weight"] = get("model.decoder.layer_norm.weight")
    t["decoder.ln.bias"] = get("model.decoder.layer_norm.bias")

    # Output projection is tied to the token embedding; Whisper has no bias.
    t["decoder.lm_head.weight"] = t["decoder.token_embedding.weight"].copy()
    t["decoder.lm_head.bias"] = zeros(cfg["n_vocab"])
    return t


def _attention(t, get, src, dst, d, zeros):
    t[f"{dst}.query.weight"] = get(f"{src}.q_proj.weight")
    t[f"{dst}.query.bias"] = get(f"{src}.q_proj.bias")
    t[f"{dst}.key.weight"] = get(f"{src}.k_proj.weight")
    t[f"{dst}.key.bias"] = zeros(d)
    t[f"{dst}.value.weight"] = get(f"{src}.v_proj.weight")
    t[f"{dst}.value.bias"] = get(f"{src}.v_proj.bias")
    t[f"{dst}.out.weight"] = get(f"{src}.out_proj.weight")
    t[f"{dst}.out.bias"] = get(f"{src}.out_proj.bias")


def _mlp(t, get, src, dst):
    t[f"{dst}.mlp.0.weight"] = get(f"{src}.fc1.weight")
    t[f"{dst}.mlp.0.bias"] = get(f"{src}.fc1.bias")
    t[f"{dst}.mlp.2.weight"] = get(f"{src}.fc2.weight")
    t[f"{dst}.mlp.2.bias"] = get(f"{src}.fc2.bias")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    hf, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(hf)
    weights = convert_weights(load_file(str(hf / "model.safetensors")), cfg)
    save_file(weights, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(load_tokenizer(hf, cfg["n_vocab"]), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(weights)} tensors and config/tokenizer to {out}")


if __name__ == "__main__":
    main()
