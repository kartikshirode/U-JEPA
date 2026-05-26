"""Phase 0 gate: reproduce LatentMAS GSM8K accuracy on Qwen3-14B-AWQ.

Calls vendored LatentMASMethod directly with our own argparse.Namespace,
bypassing run.py's argparse choices list which is hardcoded to
{Qwen3-4B, Qwen3-14B} only. Using AWQ-quantized weights so 14B fits
into 2x T4 16GB with plenty of room for KV cache and activations.

Memory budget per T4 (16 GiB total reported as 14.56):
  Qwen3-14B-AWQ weights (4-bit): ~3.5 GiB per GPU at TP=2
  Activations + KV cache budget:  ~10 GiB per GPU
  Slack:                          ~1 GiB

Gate: GSM8K accuracy >= 65% on 250 problems.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Path bootstrap for u_jepa package + vendored LatentMAS
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_VENDORED = _REPO_ROOT / "vendored" / "LatentMAS"
for p in (_SRC, _VENDORED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from u_jepa.util.env import detect, prepare

# Patch vllm.LLM to default max_model_len so vendored ModelWrapper does not
# request the full 40960-token native context (way larger than our T4 KV cache
# budget at ~10k tokens). Done at import-time so it lands before vendored
# LatentMAS imports vllm.
#
# Also force off prefix caching and force eager execution: the vendored
# latent_mas path feeds vLLM `prompt_embeds` (not token ids) and the prefix
# cache plus CUDA-graph capture combo trips an internal assertion
# `len(inputs_embeds) == len(input_tokens)` from batch 2 onward in vLLM 0.10
# (V0 engine on Turing). Disabling prefix caching is correctness-neutral here
# (the latent path embeds are unique per problem so the cache never helps),
# enforce_eager just costs throughput. max_num_seqs is pinned to the configured
# generate_bs to keep the scheduler from sizing buffers for batches we never
# send. Override rather than setdefault so vendored ModelWrapper cannot
# re-enable prefix caching by passing it explicitly.
def _patch_vllm_max_model_len(
    default_max_model_len: int = 8192,
    max_num_seqs: int = 2,
) -> None:
    try:
        import vllm  # type: ignore
    except ImportError:
        return  # local Windows path
    _orig = vllm.LLM.__init__

    def _patched(self, *args, **kwargs):
        kwargs.setdefault("max_model_len", default_max_model_len)
        kwargs["max_num_seqs"] = max_num_seqs
        kwargs["enable_prefix_caching"] = False
        kwargs["enforce_eager"] = True
        # Vendored ModelWrapper only sets enable_prompt_embeds=True on the
        # prefix-caching branch (models.py:50). With prefix caching forced off
        # vendored takes the else-branch at models.py:52 which omits the flag,
        # and vLLM then rejects prompt_embeds inputs with
        # "You must set --enable-prompt-embeds to input prompt_embeds".
        # Force it on here so vendored's branch choice no longer matters.
        kwargs["enable_prompt_embeds"] = True
        return _orig(self, *args, **kwargs)

    vllm.LLM.__init__ = _patched
    print(
        f"[patch] vllm.LLM defaults: max_model_len={default_max_model_len}, "
        f"max_num_seqs={max_num_seqs}, enable_prefix_caching=False, "
        f"enforce_eager=True, enable_prompt_embeds=True"
    )

_patch_vllm_max_model_len(default_max_model_len=8192, max_num_seqs=2)


def _patch_transformers_activations_for_autoawq() -> None:
    """autoawq imports PytorchGELUTanh / NewGELUActivation / GELUActivation
    from transformers.activations; transformers >= 4.53 dropped those class
    names in favor of an ACT2FN dict. Add thin shims so autoawq's import
    chain succeeds on Kaggle's transformers (currently 4.57)."""
    try:
        import transformers.activations as _act
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return  # local Windows path

    if not hasattr(_act, "PytorchGELUTanh"):
        class PytorchGELUTanh(nn.Module):
            def forward(self, x):
                return F.gelu(x, approximate="tanh")
        _act.PytorchGELUTanh = PytorchGELUTanh

    if not hasattr(_act, "NewGELUActivation"):
        class NewGELUActivation(nn.Module):
            def forward(self, x):
                return F.gelu(x, approximate="tanh")
        _act.NewGELUActivation = NewGELUActivation

    if not hasattr(_act, "GELUActivation"):
        class GELUActivation(nn.Module):
            def forward(self, x):
                return F.gelu(x)
        _act.GELUActivation = GELUActivation

    print("[patch] transformers.activations shimmed for autoawq compatibility")

_patch_transformers_activations_for_autoawq()


def _patch_latent_realign_to_cpu_build() -> None:
    """Vendored ModelWrapper._build_latent_realign_matrix casts the model's
    full input and output embedding weights to fp32 on the target GPU. For
    Qwen3-14B that is ~3 GB each (152K vocab x 5120 hidden) and OOMs cuda:1
    after the HF AWQ model already took ~12 GB.

    Override to do the heavy compute on CPU (one-time ~10 sec) and only
    move the resulting [hidden, hidden] matrix (~100 MB) back to the
    requested device. Behavior is otherwise identical; identity-fallback
    when latent_space_realign=False is preserved.
    """
    try:
        import sys, torch
        _vendored = Path(__file__).resolve().parents[1] / "vendored" / "LatentMAS"
        if str(_vendored) not in sys.path:
            sys.path.insert(0, str(_vendored))
        from models import ModelWrapper  # type: ignore
    except ImportError:
        return  # local Windows path

    def _patched_build(self, model, device, args):
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (input_embeds is None or output_embeds is None
                or not hasattr(input_embeds, "weight")
                or not hasattr(output_embeds, "weight")):
            raise RuntimeError("Cannot build latent realignment matrix: embeddings not accessible")
        cpu = torch.device("cpu")
        input_weight = input_embeds.weight.detach().to(device=cpu, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=cpu, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=cpu, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()
        if not getattr(args, "latent_space_realign", False):
            realign_matrix = torch.eye(realign_matrix.shape[0], device=cpu, dtype=realign_matrix.dtype)
        target_device = torch.device(device) if not isinstance(device, torch.device) else device
        return realign_matrix.to(target_device), target_norm.to(target_device)

    ModelWrapper._build_latent_realign_matrix = _patched_build
    print("[patch] _build_latent_realign_matrix compute moved to CPU to avoid cuda:1 OOM")

_patch_latent_realign_to_cpu_build()


def _patch_latent_mas_run_batch_vllm_no_pad() -> None:
    """Replace vendored run_batch_vllm to avoid the zero-pad on prompt_embeds.

    The original (vendored latent_mas.py lines 379-384) pads variable-length
    per-item embeddings to the batch max with zeros so they can be stacked.
    vLLM then treats those trailing zero rows as real prompt tokens, which
    corrupts judger accuracy on the shorter items in the batch. It also
    contributes to a scheduler assertion across batches in vLLM 0.10 because
    different problems produce different pad amounts and the internal
    `len(inputs_embeds) == len(input_tokens)` check trips.

    Our replacement keeps the variable-length list as-is (vLLM's API accepts
    a list of `{"prompt_embeds": tensor}` dicts where each tensor can have
    its own length) and does not stack into one padded tensor. Behavior is
    otherwise byte-identical to the upstream method.
    """
    try:
        import sys, torch
        _vendored = Path(__file__).resolve().parents[1] / "vendored" / "LatentMAS"
        if str(_vendored) not in sys.path:
            sys.path.insert(0, str(_vendored))
        from methods.latent_mas import LatentMASMethod  # type: ignore
        from models import _past_length  # type: ignore
        from prompts import (  # type: ignore
            build_agent_message_sequential_latent_mas,
            build_agent_message_hierarchical_latent_mas,
        )
        from utils import extract_gsm8k_answer, normalize_answer  # type: ignore
    except ImportError:
        return  # local Windows path

    def _patched_run_batch_vllm(self, items):
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv = None
        agent_traces = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        embedding_record = []

        for agent in self.agents:
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = (
                self.model.prepare_chat_batch(batch_messages, add_generation_prompt=True)
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                wrapped_prompts = (
                    [f"{p}<think>" for p in prompts] if self.args.think else prompts
                )

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(
                        self.model.tokenizer.convert_ids_to_tokens(active_ids)
                    )

                past_kv, previous_hidden_embedding = (
                    self.model.generate_latent_batch_hidden_state(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = (
                        self.latent_steps if self.latent_only else tokens_added
                    )
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[
                            :, -self.latent_steps:, :
                        ]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[
                            :, 0:0, :
                        ]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                judger_prompts = (
                    [f"{p}<think>" for p in prompts] if self.args.think else prompts
                )

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_input_ids = judger_encoded["input_ids"].to(self.model.HF_device)
                judger_attn = judger_encoded["attention_mask"].to(self.model.HF_device)

                curr_prompt_emb = self.model.embedding_layer(judger_input_ids).to(
                    self.vllm_device
                )

                assert (
                    "Qwen" in self.args.model_name or "qwen" in self.args.model_name
                ), "latent_embedding_position is only supported for Qwen models currently."

                len_of_left = []
                for p in judger_prompts:
                    cut_idx = p.find("<|im_start|>user\n")
                    left = p[: cut_idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)["input_ids"]))

                B = curr_prompt_emb.shape[0]

                # Per-item variable-length embeddings, no zero padding. vLLM
                # accepts a list of `{"prompt_embeds": [L_i, H]}` dicts; each
                # item is scheduled independently and zeros would otherwise be
                # treated as real prompt content, corrupting the judger output.
                prompt_embeds_list = []
                for i in range(B):
                    active_len = int(judger_attn[i].sum().item())
                    real_curr = curr_prompt_emb[i, :active_len, :]
                    insert_idx = min(len_of_left[i], active_len)
                    left_emb = real_curr[:insert_idx, :]
                    right_emb = real_curr[insert_idx:, :]
                    combined = torch.cat(
                        [left_emb, past_embedding[i], right_emb], dim=0
                    )
                    combined = combined.to(dtype=torch.float16)
                    prompt_embeds_list.append({"prompt_embeds": combined})

                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )
                generated_texts = [out.outputs[0].text.strip() for out in outputs]

                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )

        results = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    LatentMASMethod.run_batch_vllm = _patched_run_batch_vllm
    print(
        "[patch] LatentMASMethod.run_batch_vllm now sends per-item "
        "variable-length prompt_embeds (no zero pad) to avoid corrupting "
        "judger output and the vLLM scheduler assertion across batches"
    )

_patch_latent_mas_run_batch_vllm_no_pad()

PHASE_CFG = dict(
    method="latent_mas",
    model_name="Qwen/Qwen3-14B-AWQ",
    task="gsm8k",
    split="test",
    prompt="sequential",
    max_samples=250,
    # generate_bs MUST be >= 2: vendored LatentMASMethod.run_batch_vllm does
    # `embedding_layer(...).squeeze(0)` at line 355 which collapses [1, L, H]
    # to [L, H] when bs=1, then the next line unpacks B, L, H = .shape and
    # fails. Bump to 2 to keep the squeeze a no-op.
    generate_bs=2,
    latent_steps=4,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95,
    use_vllm=True,
    # enable_prefix_caching kept False here on purpose: vendored ModelWrapper
    # reads this flag to decide between two LLM(...) init paths. We also force
    # it off via the vllm.LLM monkey-patch above so even if vendored ignores
    # the flag, vLLM never enables the prefix cache. With prompt_embeds the
    # cache never hits anyway and the combo trips an assertion across batches.
    enable_prefix_caching=False,
    use_second_HF_model=True,
    latent_space_realign=True,
    # tensor_parallel_size=1 so vLLM lives entirely on cuda:0 and the second
    # HF model (used for latent-path hidden states) gets cuda:1 to itself.
    # AWQ Qwen3-14B is ~4.7 GB so single-GPU vLLM has plenty of room.
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    device="cuda:0",
    device2="cuda:1",
    text_mas_context_length=-1,
    think=False,
    seed=42,
)


def build_namespace() -> argparse.Namespace:
    return argparse.Namespace(**PHASE_CFG)


def run_inference(args: argparse.Namespace) -> Tuple[float, int, List[Dict], float]:
    """Load model, build method, iterate dataset, return (acc, correct, preds, elapsed)."""
    # Imports happen here so the script can `--help` cleanly without CUDA
    from models import ModelWrapper
    from methods.latent_mas import LatentMASMethod
    from data import load_gsm8k
    from utils import auto_device, set_seed

    set_seed(args.seed)
    device = auto_device(args.device)
    print(f"loading {args.model_name} via vLLM with TP={args.tensor_parallel_size}")
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    method = LatentMASMethod(
        model,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        args=args,
    )

    dataset_iter = list(load_gsm8k(split=args.split))
    if args.max_samples > 0:
        dataset_iter = dataset_iter[: args.max_samples]
    total = len(dataset_iter)
    print(f"running {total} GSM8K problems")

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    t0 = time.time()

    for item in dataset_iter:
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == total:
            results = (
                method.run_batch_vllm(batch)
                if args.method == "latent_mas" and args.use_vllm
                else method.run_batch(batch)
            )
            for res in results:
                preds.append(res)
                processed += 1
                print(
                    f"[{processed}/{total}] "
                    f"pred={res.get('prediction')} "
                    f"gold={res.get('gold')} "
                    f"ok={res.get('correct')}"
                )
            batch = []

    elapsed = time.time() - t0
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / max(1, len(preds))
    return acc, correct, preds, elapsed


def main() -> int:
    env = prepare()
    print(f"env: {env.name}, repo_root: {env.repo_root}")

    if not env.can_run_vllm:
        print(f"[skip] env={env.name} cannot run vLLM. Use kaggle/phase0/.")
        return 0

    args = build_namespace()
    print(f"config: {PHASE_CFG}")

    try:
        acc, correct, preds, elapsed = run_inference(args)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    result = {
        "phase": 0,
        "stage": "latentmas_baseline",
        "config": PHASE_CFG,
        "model": args.model_name,
        "split": args.split,
        "n_eval": len(preds),
        "accuracy": acc,
        "correct": correct,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_sample": round(elapsed / max(1, len(preds)), 2),
    }
    out_json = env.results_dir / "phase0_baseline.json"
    try:
        out_json.write_text(json.dumps(result, indent=2))
    except OSError as e:
        print(f"[warn] could not write {out_json}: {e}")
    print(json.dumps(result, indent=2))

    if acc < 0.65:
        print(f"GATE FAIL: accuracy {acc:.3f} < 0.65")
        return 2
    print(f"GATE PASS: accuracy {acc:.3f} >= 0.65")
    return 0


if __name__ == "__main__":
    sys.exit(main())
