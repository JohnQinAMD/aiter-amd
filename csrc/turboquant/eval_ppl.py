"""Standalone PPL evaluation for TurboQuant weight quantization.

Quantizes model weights on-the-fly and evaluates WikiText-103 perplexity.
Does NOT require turboquant-model package — uses turboquant_engine directly.

Usage:
    python eval_ppl.py --model Qwen/Qwen2.5-3B --bit-width 4 --n-chunks 20
    python eval_ppl.py --model Qwen/Qwen2.5-3B --baseline  # FP16 baseline
"""
import argparse, sys, os, time, math, logging, importlib.util
import torch
import torch.nn as nn

_spec = importlib.util.spec_from_file_location("tq_engine",
    "/shared_aig/john/semianalysis/sglang-amd/python/sglang/srt/layers/quantization/turboquant_engine.py")
_tq_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq_engine)
turboquant_quantize_packed = _tq_engine.turboquant_quantize_packed
turboquant_matmul_pytorch = _tq_engine.turboquant_matmul_pytorch
generate_rotation_matrix = _tq_engine.generate_rotation_matrix
get_codebook = _tq_engine.get_codebook
unpack_4bit = _tq_engine.unpack_4bit
clear_rotation_cache = _tq_engine.clear_rotation_cache

logger = logging.getLogger("eval_ppl")


class TurboQuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear with TurboQuant on-the-fly dequant."""

    def __init__(self, original: nn.Linear, bit_width: int = 4, group_size: int = 128, seed: int = 42):
        super().__init__()
        W = original.weight.data.float().cpu()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.group_size = group_size
        self.bit_width = bit_width

        if group_size is None or group_size > self.in_features:
            group_size = self.in_features
        self.group_size = group_size

        tq = turboquant_quantize_packed(W, bit_width=bit_width, group_size=group_size, seed=seed)
        self.register_buffer("indices_packed", tq["indices_packed"])
        self.register_buffer("codebook", tq["codebook"])
        self.register_buffer("weight_norms", tq["norms"])
        self.seed = tq["seed"]
        self.group_size = tq["group_size"]

        if original.bias is not None:
            self.register_buffer("bias", original.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clear_rotation_cache()
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        y = turboquant_matmul_pytorch(
            x_2d, self.indices_packed, self.codebook,
            self.weight_norms, self.in_features, self.group_size, self.seed
        )
        y = y.reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)


def quantize_model_inplace(model, bit_width=4, group_size=128, seed=42,
                           skip_embeddings=True, skip_lm_head=True):
    """Replace nn.Linear layers with TurboQuantLinear."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if skip_embeddings and ("embed" in name or "wte" in name or "wpe" in name):
                continue
            if skip_lm_head and ("lm_head" in name or "head" in name):
                continue

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            tq_linear = TurboQuantLinear(module, bit_width, group_size, seed + count)
            tq_linear = tq_linear.to(module.weight.device)
            setattr(parent, child_name, tq_linear)
            count += 1

    logger.info(f"Quantized {count} linear layers to {bit_width}-bit")
    return model


def eval_ppl(model, tokenizer, seq_length=512, n_chunks=50, device="cuda"):
    """Evaluate perplexity on WikiText-103 validation set."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    n_chunks = min(n_chunks, (len(input_ids) - 1) // seq_length)
    logger.info(f"Evaluating {n_chunks} chunks of {seq_length} tokens")

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            start = i * seq_length
            chunk = input_ids[start : start + seq_length + 1].unsqueeze(0).to(device)
            outputs = model(chunk[:, :-1])
            logits = outputs.logits
            targets = chunk[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

            if (i + 1) % 10 == 0:
                running_ppl = math.exp(total_loss / total_tokens)
                logger.info(f"  Chunk {i+1}/{n_chunks}: running PPL = {running_ppl:.4f}")

    ppl = math.exp(total_loss / total_tokens)
    return ppl


def main():
    parser = argparse.ArgumentParser(description="TurboQuant PPL evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="HF model name")
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--n-chunks", type=int, default=50)
    parser.add_argument("--baseline", action="store_true", help="Evaluate FP16 baseline only")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.baseline:
        logger.info("Evaluating FP16 baseline...")
        ppl = eval_ppl(model, tokenizer, args.seq_length, args.n_chunks)
        print(f"\nFP16 Baseline PPL: {ppl:.4f}")
    else:
        # First evaluate baseline
        logger.info("Evaluating FP16 baseline...")
        ppl_base = eval_ppl(model, tokenizer, args.seq_length, args.n_chunks)
        print(f"\nFP16 Baseline PPL: {ppl_base:.4f}")

        # Then quantize and evaluate
        logger.info(f"Quantizing to {args.bit_width}-bit (group_size={args.group_size})...")
        t0 = time.time()
        model = quantize_model_inplace(model, args.bit_width, args.group_size, args.seed)
        quant_time = time.time() - t0
        logger.info(f"Quantization took {quant_time:.1f}s")

        logger.info("Evaluating TurboQuant PPL...")
        ppl_tq = eval_ppl(model, tokenizer, args.seq_length, args.n_chunks)
        print(f"TurboQuant {args.bit_width}-bit PPL: {ppl_tq:.4f}")
        print(f"PPL degradation: {ppl_tq - ppl_base:+.4f}")


if __name__ == "__main__":
    main()
