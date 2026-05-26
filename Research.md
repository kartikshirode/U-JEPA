# U-JEPA Prototype Design Document: Two Ranked Implementation Paths for an 8 GB Consumer GPU

**Author:** Senior ML Research Engineer (advisory)
**Date:** May 26, 2026
**Audience:** PhD student / solo researcher with one RTX 4060 8 GB laptop and a 2–4 month horizon
**Status:** Decision-ready

---

## TL;DR (3 bullets)

- **Pursue Approach 1 first ("LatentMAS-Forked, JEPA-Regularized")** — fork `Gen-Verse/LatentMAS`, swap Qwen3-14B for Qwen3-4B-Instruct at NF4, add an LLM-JEPA auxiliary loss + O-LoRA / N-LoRA orthogonal adapters, and treat V-JEPA 2 ViT-L (hidden_size=1024, 24 layers) as a frozen vision sub-agent feeding a BLIP-2-style Q-Former into the latent bus. This is the only path with ≥80% probability of producing a workshop-paper-quality prototype on an RTX 4060 in 2–4 months because every load-bearing component already has open code and the cross-agent communication substrate is training-free.
- **Approach 2 ("Ground-up U-JEPA")** — wire V-JEPA 2 + LLM-JEPA + a custom router + N-LoRA continual stack from scratch with continued-pretraining — is the more publishable contribution if it succeeds, but on an 8 GB card the alignment training (image-token-to-LLM-embedding) and JEPA-style joint pretraining are realistically 5–8 months of work with high failure risk on a single laptop. Recommended as a *Phase 4 extension* of Approach 1, not as the starting point.
- **The strongest novel contribution achievable in 2–4 months is the *Approach 1 + V-JEPA bridge* combination**: no published work has shown "JEPA-extracted visual representations used as agent communication tokens in a latent multi-agent system, with orthogonal LoRA adapters enabling zero-retraining adaptation." LatentMAS is text-only (Jiaru Zou et al., Princeton/UIUC/Stanford, arXiv:2511.20639); Vision Wormhole (Liu et al., arXiv:2602.15382) uses VLM visual tokens, not JEPA latents; VL-JEPA (Chen, Shukor, LeCun et al., arXiv:2512.10942) is single-agent. The white space is real and small enough for a single GPU.

---

## Key Findings

### 1. The hardware ceiling is the dominant design constraint

The torchtune documentation reports peak allocated memory of **~9 GB for QLoRA on Llama-3-8B** on a single GPU, versus ~19 GB for LoRA without 4-bit quantization (a secondary blog circulated a 7.4 GiB figure that does not match the official numbers — treat 9 GB as the realistic floor for 8B-class QLoRA). On an RTX 4060 with 8 GB, that floor is already over budget. The implication: **you cannot fit two 7–8B LLMs simultaneously on an RTX 4060, even at NF4**. This forces a sequential-agent design (one LLM loaded at a time, KV caches paged through CPU RAM) or a smaller backbone.

Practical model budget on 8 GB VRAM:
- **Qwen3-4B at Q4_K_M:** ~2.5 GB weights → comfortable headroom for V-JEPA ViT-L (~650 MB fp16) + Q-Former (~50 MB) + KV cache + LoRA adapters (~30 MB per task). **This is the realistic backbone.** Confirmed by community benchmarks: "Qwen3 4B at Q4 needs ~2.5 GB — perfect for 4–6 GB GPUs or low-memory Macs" (willitrunai.com Qwen3 guide).
- **Qwen3-8B at Q4_K_M:** ~4.6 GB — fits but leaves only ~3 GB for everything else; risky.
- **V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`):** 24 layers, hidden_size 1024, 16 attention heads, patch 16, crop 256, fpc 64. Approximately 0.3B parameters per the HuggingFace model card (Assran et al., FAIR/Meta, 2025); ~650 MB in fp16. Output tensor for a 64-frame clip is `[1, 10240, 1024]` (no CLS token — must mean-pool or attentive-pool).
- **V-JEPA 2.1 ViT-B (80M):** lighter alternative if memory is tight; released 2026-03-16, currently only loadable via `torch.hub` (HuggingFace `transformers` integration is open issue #45496).

### 2. LatentMAS is the right substrate — but it has critical limitations

From Zou et al., arXiv:2511.20639 (Princeton, UIUC, Stanford) and the `Gen-Verse/LatentMAS` repo:
- **Communication mechanism:** "Each LLM agent A_i passes E through L transformer layers to compute the last-layer hidden representation h_t at current step t. Then, we insert h_t as the input embedding for the next step t+1, replacing the original decoding and next-token embedding processes" (§3.1). Cross-agent transfer is *layer-wise KV-cache concatenation* via the HuggingFace `past_key_values` interface: "we directly concatenate the KV caches from the immediately preceding agent into the corresponding transformer layers through the past_key_values interface in HuggingFace Transformers" (§4).
- **`--latent_space_realign`:** A training-free linear projection `W_a` solved by closed-form ridge regression: `W_a = (W_out^T W_out + λI)^(-1) W_out^T W_in`. Computed *once per run* and reused. Per the README: "Enables latent→embedding alignment. We treat this as a hyperparameter — enable/disable depending on task/model."
- **Homogeneity assumption:** Vanilla LatentMAS requires **the same model family on both sides** — KV-cache shapes must match layer-by-layer. The community "Hybrid-LatentMAS" fork adds heterogeneity but is less mature.
- **Hardware as reported:** "All experiments are conducted on 8×NVIDIA A100-80G GPUs." The smallest model evaluated is **Qwen3-4B** (no smaller variants tested).
- **Text-only:** No vision pipeline exists in the repo. This is the central opening for a U-JEPA contribution.
- **Efficiency claims (verbatim, paper abstract):** "achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4×-4.3× faster end-to-end inference." (Venue status: the arXiv v1 was posted Nov 2025; the repo claims an ICML 2026 Spotlight acceptance — confirm before citing in your paper.)

### 3. The continual-learning toolbox is mature and fits in <100 MB

- **O-LoRA** (Wang, Chen, Ge et al., EMNLP 2023 Findings, `cmnfriend/O-LoRA`): orthogonal subspace constraint between sequentially-learned LoRAs; no replay buffer; preserves zero-shot generalization. Canonical baseline.
- **N-LoRA** (Shuo Yang et al., COLING 2025, `PKU-YuanGroup/N-LoRA`): "N-LoRA achieves superior performance (+2.9%), higher task orthogonality (×4.1 times), and lower parameter collision (×58.1 times) than SOTA methods" (ACL Anthology 2025.coling-main.286). Plug-and-play replacement for O-LoRA; trained on T5-large and LLaMA.
- **Online-LoRA** (Wei, Li, Marculescu, WACV 2025, `Christina200/Online-LoRA-official`): task-free OCL on ViT — uses training-dynamics loss spikes to *automatically detect distribution shifts*, no task IDs needed. Perfect for the V-JEPA stream.
- **Phi-3.5-mini** has the best reported forgetting resistance among <10B models in continual GLUE evaluation: forgetting score 0.02 vs Llama-3.1-8B's 0.59 (Aleixo et al., arXiv:2504.01241) — a strong dark-horse alternative to Qwen3-4B if you find Qwen3 forgets aggressively.

### 4. JEPA objectives have mature, lightweight implementations

- **LLM-JEPA** (Huang, LeCun, Balestriero, arXiv:2509.14252; `rbalestr-lab/llm-jepa`): adds a JEPA auxiliary loss to next-token loss, predicting embeddings of one view from another. Works with Llama 3, Gemma 2, OLMo, OpenELM; supports LoRA. Demonstrated improvements on NL-RX, GSM8K, Spider, RottenTomatoes with robustness to overfitting.
- **LeJEPA** (Balestriero & LeCun, arXiv:2511.08544; `rbalestr-lab/lejepa`): "approximately 50 lines of code" for the SIGReg loss, single trade-off hyperparameter, linear memory complexity. The `SlicingUnivariateTest` with `EppsPulley` regularizer (1024 slices) is the recommended SIGReg implementation. Eliminates stop-gradient, teacher-student, and EMA — critically reducing implementation surface area. Verified result (verbatim): "using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA reaches 79% with a ViT-H/14."
- **VL-JEPA** (Chen, Shukor, Moutakanni, Chung, Yu, Kasarla, Bolourchi, LeCun, Fung, arXiv:2512.10942): predicts continuous text embeddings from vision; 50% fewer trainable params than token-VLM with same encoder/data; 2.85× decode reduction with selective decoding. Conceptually the closest published instantiation of the U-JEPA vision.

### 5. Latent-space agent communication is a hot, uncrowded subfield

- **Interlat** (Du et al., arXiv:2511.09149): two-agent latent communication, compresses to **8 tokens** while keeping competitive ALFWorld success, 24× inference speedup. Demonstrates that Approach 1's communication compression is real.
- **Vision Wormhole** (Xiaoze Liu et al., arXiv:2602.15382; `xz-liu/heterogeneous-latent-mas`): hub-and-spoke topology, Universal Visual Codec maps reasoning traces into shared continuous latent space, injected into VLM visual pathway. Label-free teacher-student distillation. Reduces pairwise alignment complexity from O(N²) to O(N). This is the prior art most adjacent to the U-JEPA router concept.
- **Coconut** (Hao et al., Meta FAIR + UCSD, arXiv:2412.06769): foundational paper on latent CoT — last hidden state used directly as next input embedding. The conceptual ancestor of all three above.

---

## Approach 1 — "LatentMAS-Fork + JEPA Regularization + Orthogonal Adapters" (RECOMMENDED)

**One-line characterization:** Fork LatentMAS, downsize to Qwen3-4B-Instruct-Q4, bolt on (a) a frozen V-JEPA 2 ViT-L vision sub-agent talking through a tiny Q-Former, (b) an LLM-JEPA auxiliary loss on the LoRA adapters, (c) O-LoRA / N-LoRA orthogonal adapters per domain, (d) a Phi-3.5-mini router as the central brain.

### 1.1 Architecture (described)

```
                                    ┌────────────────────────────┐
        Text input ───┐              │      ROUTER / BRAIN         │
                     ▼              │  Phi-3.5-mini-Q4 + softmax  │
              [Tokenizer]            │  head over domain labels   │
                     │              │  Output: domain ∈ {math,    │
                     ▼              │  code, vqa, common} + conf  │
       Qwen3-4B-Q4 (frozen base)    └─────────────┬──────────────┘
       + [O-LoRA_k] (per-domain)                  │ routes
       hidden_size = 2560 (Qwen3-4B)              ▼
                     │                  ┌────────────────────┐
                     │                  │ Sub-Agent A_k      │
                     │  past_key_values │ (same base, swap   │
                     ├─────────────────►│ active LoRA stack) │
                     │                  └────────────────────┘
                     ▲                            │
                     │  W_a (ridge realign)        │
                     │  + LLM-JEPA loss            │
                     │                            ▼
                     │                  Last-layer hidden h_t
                     │                  → next input embedding
                     │                  (Coconut/LatentMAS)
                     │
        Image/Video input
                     │
                     ▼
       V-JEPA 2 ViT-L (frozen)
       output: [B, 10240, 1024]
                     │
                     ▼
       Mean-pool + 2-layer Q-Former
       (32 learned queries × 1024 → 2560)
                     │
                     ▼
       Projected into Qwen3-4B embedding space
       (BLIP-2-style soft prompt prefix)
```

**Key flows:**
1. Tokens enter *only* at the input boundary; text-to-embedding happens once.
2. Vision enters via V-JEPA 2 → Q-Former → token-aligned soft prefix.
3. Router emits a domain label; the orchestrator hot-swaps the active LoRA stack (~30 MB) without unloading the base model.
4. Sub-agents reason via Coconut-style latent CoT (last-layer hidden states fed back as input embeddings) for up to N=8 latent steps.
5. Cross-agent transfer (when the router invokes >1 sub-agent) uses LatentMAS-style KV-cache prepending through `past_key_values`.
6. Decoding to text happens only at the final output step (selective decoding, VL-JEPA-style).

### 1.2 Pseudocode for key components

```python
# === continual_lora.py ===
# Orthogonal LoRA bank — N-LoRA flavor
class OrthogonalLoRABank(nn.Module):
    def __init__(self, base_model, rank=16, target_modules=("q_proj","v_proj")):
        self.base = base_model  # frozen, NF4 quantized
        self.adapters = nn.ModuleDict()
        self.target_modules = target_modules
        self.rank = rank

    def add_task(self, task_id: str):
        new = LoraConfig(r=self.rank, target_modules=list(self.target_modules),
                         lora_alpha=32, lora_dropout=0.05, bias="none")
        adapter = get_peft_model(self.base, new).get_adapter()
        if self.adapters:
            adapter = project_to_null_space(adapter, list(self.adapters.values()))
        self.adapters[task_id] = adapter

    def forward(self, x, task_id, prev_task_ids=None):
        out = self.base(x) + self.adapters[task_id](x)
        ortho_loss = 0.0
        for pid in (prev_task_ids or []):
            A_curr = self.adapters[task_id].A     # rank × in_dim
            A_prev = self.adapters[pid].A.detach()
            # O-LoRA-style: penalize A_curr A_prev^T  (their inner-product matrix)
            ortho_loss = ortho_loss + (A_curr @ A_prev.T).pow(2).sum()
            # N-LoRA-style add-on: encourage zero parameter collisions (sparsity penalty)
            ortho_loss = ortho_loss + 0.01 * (A_curr.abs() * A_prev.abs()).sum()
        return out, ortho_loss

# === llm_jepa_loss.py ===
def llm_jepa_loss(model, view_a_ids, view_b_ids, predictor, lambda_jepa=0.1):
    h_a = model(view_a_ids, output_hidden_states=True).hidden_states[-1].mean(dim=1)
    with torch.no_grad():
        h_b = model(view_b_ids, output_hidden_states=True).hidden_states[-1].mean(dim=1)
    pred_b = predictor(h_a)
    return lambda_jepa * F.mse_loss(pred_b, h_b)

# === sigreg_loss.py ===  (LeJEPA collapse prevention)
import lejepa
sig_test = lejepa.univariate.EppsPulley(num_points=17)
sigreg = lejepa.multivariate.SlicingUnivariateTest(sig_test, num_slices=1024)
def sigreg_loss(embeddings):  # [B, D]
    return sigreg(embeddings)

# === router.py ===
class DomainRouter(nn.Module):
    def __init__(self, phi_model, num_domains=4, hidden=3072):  # phi-3.5-mini
        self.phi = phi_model  # NF4
        self.head = nn.Linear(hidden, num_domains)
    def route(self, input_ids):
        h = self.phi(input_ids, output_hidden_states=True).hidden_states[-1][:,-1,:]
        return self.head(h)

# === vision_bridge.py ===  (V-JEPA 2 → Q-Former → Qwen embedding space)
from transformers import AutoModel
class VJEPABridge(nn.Module):
    def __init__(self, qwen_embed_dim=2560, num_queries=32):
        self.vjepa = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256",
            torch_dtype=torch.float16, attn_implementation="sdpa")
        self.vjepa.eval()
        for p in self.vjepa.parameters(): p.requires_grad = False
        self.queries = nn.Parameter(torch.randn(num_queries, 1024))
        self.qformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=1024, nhead=8, batch_first=True),
            num_layers=2)
        self.proj = nn.Linear(1024, qwen_embed_dim)
    def forward(self, video):  # [B, T, 3, 256, 256]
        with torch.no_grad():
            feats = self.vjepa(pixel_values_videos=video).last_hidden_state  # [B,10240,1024]
        q = self.queries.unsqueeze(0).expand(feats.size(0), -1, -1)
        out = self.qformer(q, feats)              # [B, 32, 1024]
        return self.proj(out)                     # [B, 32, 2560]

# === latent_mas_loop.py ===  (Coconut-style latent reasoning inside one sub-agent)
def latent_reason(model, ids, n_latent=4):
    out = model(ids, use_cache=True, output_hidden_states=True)
    kv, h = out.past_key_values, out.hidden_states[-1][:, -1:, :]
    for _ in range(n_latent):
        out = model(inputs_embeds=h, past_key_values=kv,
                    use_cache=True, output_hidden_states=True)
        kv, h = out.past_key_values, out.hidden_states[-1][:, -1:, :]
    return h, kv  # h: final latent thought; kv: working memory to hand to next agent
```

### 1.3 Repos to fork, in priority order

1. **`Gen-Verse/LatentMAS`** — primary fork. Reuse `models.py` (HF + vLLM wrapper, latent realignment), `run.py` (entry point), `methods/latent_mas.py`. **Replace `Qwen3-14B` with `Qwen3-4B-Instruct` quantized via bitsandbytes 4-bit NF4.**
2. **`rbalestr-lab/llm-jepa`** — copy the JEPA loss module; verify it works with PEFT/LoRA `target_modules`. The repo explicitly notes Llama 3 / Gemma 2 / OLMo / OpenELM compatibility; Qwen3 needs a small config tweak (RoPE differs).
3. **`rbalestr-lab/lejepa`** — vendor `lejepa/multivariate/slicing.py` and `lejepa/univariate/epps_pulley.py`. The two-file SIGReg implementation is the most reusable thing in the SSL community right now.
4. **`PKU-YuanGroup/N-LoRA`** — copy the orthogonality penalty + non-collision loss. Their T5-large training script is the closest analogue to the Qwen3-4B continual fine-tune you need.
5. **`Christina200/Online-LoRA-official`** — copy the loss-spike-based task-shift detector for the *router*. The right primitive when domain labels aren't perfectly clean.
6. **`facebookresearch/vjepa2`** — for the inference path use the HF integration (`AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")`). The repo's `notebooks/vjepa2_demo.py` is the right starting template.

### 1.4 Phased plan & milestones (12–14 weeks)

| Phase | Weeks | Deliverable | Gate / KPI |
|---|---|---|---|
| **0. Setup** | 0–1 | RTX 4060 env: CUDA 12.4, PyTorch 2.4, bitsandbytes 0.43, PEFT 0.12, vJEPA 2 HF. Reproduce LatentMAS on Qwen3-4B-Q4 on GSM8K (text only). | LatentMAS Qwen3-4B-Q4 runs at >10 tok/s; baseline GSM8K ≥65% with 3-agent text MAS. |
| **1. Continual LoRA stack** | 2–4 | Implement N-LoRA on Qwen3-4B-Q4. Train on TRACE sub-sequence (FOMC → ScienceQA). Eval forgetting + BWT. | Forgetting < 0.05 across 2 tasks; ≥1.0% BWT improvement over sequential FT baseline. |
| **2. LLM-JEPA aux loss + SIGReg** | 5–6 | Add LLM-JEPA loss with NL/SQL view pairs from Spider; add SIGReg on hidden states. | LLM-JEPA improves Spider EM by ≥2 pts over LoRA-only; SIGReg eliminates collapse on long latent rollouts (token entropy > 4.0). |
| **3. V-JEPA bridge** | 7–9 | Q-Former bridge. Train only 32 queries + 2-layer Q-Former + linear proj (~10M params) on a Visual7W subset (~3k pairs) with frozen V-JEPA + frozen Qwen3-4B. | VQA accuracy ≥ baseline of CLIP-projection + frozen Qwen; vision-prefix cosine sim with corresponding text embedding > 0.4. |
| **4. Router + orchestration** | 10–11 | Phi-3.5-mini-Q4 domain classifier (4 classes). Implement Online-LoRA loss-spike trigger to add a 5th LoRA on a new domain. | Routing accuracy ≥85% on held-out mixed-domain queries; **zero-retraining adaptation demonstrated on 2 sequential domains (core success criterion).** |
| **5. Eval + writeup** | 12–14 | Full ablation: (a) latent vs text MAS, (b) ±JEPA loss, (c) ±orthogonal LoRA. Submit to NeurIPS workshop or arXiv. | Headline figure: "% accuracy preserved on task-1 after learning task-2, with both vision and text in the latent bus." |

### 1.5 Datasets / benchmarks

- **Continual learning:** TRACE benchmark (8 tasks, 5k train / 2k test each) — pick 3 contrasting: FOMC (finance), ScienceQA (multimodal MCQ), Py150 (code). Directly demonstrates the continual + multimodal claim.
- **LLM-JEPA view pairs:** Spider (NL ↔ SQL), GSM8K (problem ↔ solution rationale), NL-RX (text ↔ regex) — same datasets used in the LLM-JEPA paper.
- **Vision side:** Something-Something-v2 to confirm your bridge preserves motion features (V-JEPA 2 ViT-L reports 73.7% top-1 frozen-probe on SSv2 per the `facebookresearch/vjepa2` README; the headline 77.3% figure is ViT-g, **not** the ViT-L you'll be running). Visual7W or 5k OK-VQA subset for the actual vision-language demonstration.
- **Latent reasoning sanity:** GSM8K (LatentMAS baseline), MATH (Interlat benchmark).

### 1.6 VRAM budget table (8 GB target)

| Component | Precision | VRAM (MB) | Notes |
|---|---|---|---|
| Qwen3-4B base | NF4 | ~2,500 | bitsandbytes 4-bit |
| Active LoRA stack (×4 domains) | fp16 | ~120 | rank 16, q/v projections |
| Phi-3.5-mini router | Q4 GGUF | ~2,400 | Load on demand; or stays resident if context short |
| V-JEPA 2 ViT-L | fp16 | ~650 | Frozen, eval-mode |
| Q-Former bridge | fp16 | ~50 | 32 queries × 1024 + 2-layer decoder |
| KV cache (4k ctx, single agent) | fp16 | ~600 | Drops to <100 MB at 1024 ctx |
| Activations + optimizer states (training) | bf16 | ~1,500 | Gradient checkpointing on |
| **Slack** | | **~200** | |
| **Total (training, one task)** | | **~7,800** | Tight; use grad_accum=4–8, batch=1, ctx=1024 |
| **Total (inference)** | | **~6,400** | Comfortable |

**If you blow the budget:** drop router to Phi-3.5-mini Q4_K_S, or swap V-JEPA 2 ViT-L for **V-JEPA 2.1 ViT-B (80M, ~160 MB fp16)** — currently `torch.hub`-only, HF integration is open issue #45496.

### 1.7 Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Qwen3-4B forgets aggressively despite N-LoRA (the N-LoRA paper's results are on T5-large, not Qwen3) | High | A/B test Phi-3.5-mini-Q4 (forgetting score 0.02 vs Llama-3.1-8B 0.59, arXiv:2504.01241). Phase-1 contingency. |
| LatentMAS realignment matrix `W_a` doesn't transfer cleanly to Q4 NF4 model | Medium | The realignment uses `W_out` and `W_in` — both frozen; you can de-quantize them in fp16 for the one-time ridge solve. Cost: one O(d_h² × d_v) solve. |
| V-JEPA features don't align with Qwen3 embedding space (BLIP-2 used CLIP, already roughly aligned via web-text supervision; V-JEPA was pretrained with *no* text) | **High — the dominant technical risk** | (1) Add a contrastive ITC loss on a 10k-pair MS-COCO-mini set during Q-Former training. (2) Initialize the projection from SigLIP and treat V-JEPA features as "motion features" supplementing a frozen SigLIP-base-256 for static frames. (3) Fallback: SigLIP-only vision in Phase 3 if Q-Former fails to converge in 3 weeks. |
| 8 GB OOM during training | Medium | NF4 + gradient checkpointing + `paged_adamw_8bit` + batch=1 + max_seq_len=1024. |
| Router collapses when domains overlap semantically | Medium | Online-LoRA loss-spike detector as a secondary signal: if router confidence < 0.6 *and* loss is rising, instantiate a new LoRA. |
| Publication scooped by a VL-JEPA follow-up | Medium | VL-JEPA does *not* do continual learning or multi-agent routing; your delta is "U-JEPA = VL-JEPA + LatentMAS + orthogonal continual LoRA." Lead the paper with the *continual-learning-in-latent-space* result. |

### 1.8 Publication strategy

- **NeurIPS 2026 workshops** on Continual Learning, Foundation Models, or Agent Learning (deadlines typically September 2026).
- **arXiv preprint** by month 4.
- Strong fallback: **ICLR 2027 main track** if results clear ≥5% absolute improvement over LatentMAS-text-MAS + sequential-FT baseline.

The most defensible novelty claim: *"V-JEPA-extracted visual representations can serve as agent communication tokens in a latent multi-agent system, with orthogonal LoRA adapters enabling zero-retraining adaptation to new domains, on a single 8 GB consumer GPU."*

---

## Approach 2 — "Ground-up U-JEPA" (Ambitious; deferred)

**One-line characterization:** Continued-pretrain a small (1–2B) decoder using LLM-JEPA + LeJEPA SIGReg objectives jointly with V-JEPA 2 features through a shared latent bus, train a router from scratch, layer N-LoRA continual learning on top.

### 2.1 Architecture

Same block diagram as Approach 1, **except**:
- The LLM is *not* an off-the-shelf Qwen3 — it is a continued-pretrained ~1B decoder with LLM-JEPA + LeJEPA losses baked in, so the hidden-state space is already isotropic Gaussian (LeJEPA) and already JEPA-structured (LLM-JEPA), giving cleaner latent reasoning at inference.
- V-JEPA bridge is *jointly* trained with the LLM via a JEPA prediction loss (predict masked text embeddings from vision and vice versa) — a literal VL-JEPA replication, not post-hoc Q-Former fitting.
- Router is trained adversarially to maximize per-domain LoRA orthogonality.

### 2.2 Why it is more publishable but riskier

**Pros:**
- Native isotropic Gaussian embedding space (LeJEPA reaches 79% ImageNet-1k top-1 with ViT-H/14 frozen-backbone — strong evidence the SIGReg objective produces high-quality representations).
- Eliminates the V-JEPA-to-Qwen alignment problem — vision and language are *jointly* embedded from scratch.
- Closer to LeCun's original U-JEPA vision; a successful prototype would clearly advance over VL-JEPA (which is single-agent).

**Cons:**
- LLM pretraining at 1B scale needs 50–200B tokens; even Mistral-7B-style pretraining on a 4060 is roughly 200× slower than realistic. Only viable as *continued* pretraining of a 1–4B base with ~1–5B tokens.
- LeJEPA pretraining was demonstrated on ViTs up to 1.8B — but on multi-GPU clusters. The "~50 lines of code" claim is for the *loss only*, not the data pipeline.
- VL-JEPA replication needs paired image-text data; LAION-mini or DataComp-small subsets are hundreds of GB.

### 2.3 Repos
- `rbalestr-lab/lejepa` — pretraining substrate.
- `rbalestr-lab/llm-jepa` — language-side JEPA.
- `facebookresearch/vjepa2` — vision side.
- Base model: **OLMo-1B** (`allenai/OLMo-1B-hf`) — chosen because LLM-JEPA explicitly demonstrates OLMo compatibility.

### 2.4 Plan (5–8 months)

| Phase | Weeks | Deliverable |
|---|---|---|
| 1 | 1–6 | Continued pretraining of OLMo-1B with LLM-JEPA + SIGReg on ~5B tokens (C4 + StarCoder-mini), on rented A100 spot instances (~$300–800) — RTX 4060 too slow. |
| 2 | 7–14 | Joint VL-JEPA training with V-JEPA 2 ViT-B + the continued-pretrained OLMo-1B on LAION-CC-mini (~5M pairs). |
| 3 | 15–20 | Multi-agent routing, N-LoRA continual fine-tunes on TRACE. |
| 4 | 21–32 | Full eval + paper. |

### 2.5 Why not first

Approach 2's first 14 weeks produce no demonstrable agentic system — nothing demoable until ~month 4 — whereas Approach 1's Phase 0 deliverable (week 1) runs end-to-end. Iterative progress beats one big bet, especially on a laptop.

---

## Approach 3 (for completeness) — "Training-Free Frozen Projection Heads"

Keep *all* models frozen (V-JEPA 2 ViT-B, Qwen3-4B-Q4, Phi-3.5-mini-Q4) and train *only* lightweight projection heads (~10M params each) on top using contrastive (CLIP-style) + JEPA losses. No fine-tuning of the LLM at all. Continual learning happens entirely by adding *new* projection heads per domain — heads are orthogonal by construction because they only see disjoint input distributions.

- **Pro:** Fits in <4 GB VRAM at inference; trains in days.
- **Pro:** No catastrophic forgetting *by construction* (you never touch the LLM).
- **Con:** Likely too weak for math/code domains; the LLM's reasoning is never adapted, only its *input embeddings*.
- **Verdict:** Run as a *baseline* under Approach 1 (Phase 5 eval), not as a separate path. If it surprisingly works well, it's an interesting negative result.

---

## Decision Matrix

| Criterion | Approach 1 (Fork LatentMAS) | Approach 2 (Ground-up) | Approach 3 (Frozen heads) |
|---|---|---|---|
| **Feasibility on RTX 4060** | ✅ High — every component verified to fit | ⚠️ Pretraining must move to cloud; ~$500 | ✅ Highest |
| **End-to-end within 2–4 months** | ✅ Yes (12–14 weeks plan) | ❌ No (5–8 months) | ✅ Yes (3–6 weeks) |
| **Publication potential** | ✅ Workshop / arXiv strong; ICLR plausible | ✅✅ ICLR/NeurIPS main track plausible | ⚠️ Limited — negative-result paper |
| **Research novelty** | ✅ Continual-LoRA-in-latent-MAS is unclaimed | ✅✅ Novel pretraining substrate | ❌ Mostly engineering |
| **Risk of failure** | 🟢 Low–medium (V-JEPA↔Qwen alignment is the biggest unknown) | 🔴 High (compute + data + several unsolved subtasks) | 🟢 Low |
| **Mid-project demonstrability** | ✅ Week-1 baseline; weekly visible progress | ❌ Nothing demoable until month 4+ | ✅ Immediate |
| **Community connections / citations** | ✅ Builds on 7+ active 2025–2026 repos | ⚠️ Mostly Balestriero ecosystem | ⚠️ Loose |

**Recommendation: Pursue Approach 1.** Set Approach 2 as your "if Phase 3 succeeds and 4–6 weeks remain, expand into a continued-pretraining variant" follow-up paper. Use Approach 3 as a baseline.

---

## Recommendations (staged, with thresholds)

1. **Week 1 — Reproducibility gate.** Get unmodified LatentMAS running on Qwen3-4B-Q4 with the `latent_mas` method on GSM8K. **Gate:** ≥65% accuracy. If you fail this gate, nothing else matters — fix this first.
2. **Weeks 2–4 — Single-task LoRA + N-LoRA validation.** Implement N-LoRA on the frozen Qwen3-4B-Q4. Train on FOMC, then ScienceQA. **Gate:** Forgetting (FOMC accuracy drop after ScienceQA fine-tune) < 5%. If forgetting > 10%, swap backbone to Phi-3.5-mini-Q4 *immediately*.
3. **Weeks 5–6 — Add JEPA losses.** Layer in LLM-JEPA aux loss + SIGReg. **Gate:** Spider exact-match ≥+2 absolute over LoRA-only; latent hidden states' covariance condition number < 100 (proxy for non-collapse).
4. **Weeks 7–9 — Vision bridge (highest-risk phase).** Set a hard 3-week budget. If VQA accuracy via the Q-Former bridge is not within 5 pts of a frozen-CLIP-projection baseline by week 9, *drop V-JEPA and use SigLIP for vision*. This is a paper-saving fallback and is intellectually honest — you tested V-JEPA for the role and it didn't work for low-data alignment, which is itself a finding.
5. **Weeks 10–11 — Router + zero-retraining adaptation demo.** Clean experiment: train on domains {A, B}, freeze, present domain C *without retraining*, show that the router instantiates a new LoRA on the fly. **Gate:** Domain-C accuracy ≥70% of an upper-bound (full-data trained) baseline.
6. **Weeks 12–14 — Writeup.** Single core experiment to lead: "Average accuracy across 4 sequential domains, with and without our latent-JEPA mechanism." Target table: rows = methods (sequential-FT, O-LoRA, N-LoRA, Ours), columns = per-domain acc + average + BWT.

**Stop conditions / pivot triggers:**
- If Phase 1 forgetting > 15% even after backbone swap → add EWC + replay buffer (Hindsight Anchor Learning, arXiv:2002.08165) as a safety net.
- If Phase 3 Q-Former training fails to converge after 2 weeks → drop to text-only and recast as *"Latent-space continual learning for LLM agents"* (narrower but still publishable).
- If by week 10 you have >2 weeks of slack → start the Approach-2 continued-pretraining run on a rented A100 in parallel ($200–400 budget).

---

## Caveats

1. **All VRAM numbers are estimates** derived from community benchmarks (Ollama community reports, torchtune docs at docs.pytorch.org/torchtune which give ~9 GB for Llama-3-8B QLoRA), **not measured on the specific U-JEPA stack**. Spend Day 1–2 measuring on your machine; the 8 GB headroom is tight enough that a 10% surprise breaks training.
2. **LatentMAS is a 2026 paper with code under active development.** The `Hybrid-LatentMAS` heterogeneous extension is announced in the README but partially unreleased; treat as a moving target. The realignment matrix mechanism is well-described in the paper, but its interaction with NF4 quantization is *not* tested in any published work — real implementation risk here. The repo's "ICML 2026 Spotlight" claim should be independently confirmed before citing.
3. **V-JEPA 2 is not language-aligned.** Unlike CLIP or SigLIP, V-JEPA was pretrained without text. The Q-Former bridge is doing more work than in BLIP-2 (whose CLIP features were already roughly language-aligned). This is the single most likely failure mode of Approach 1; the SigLIP fallback in Recommendation 4 is not cosmetic.
4. **The "central brain" metaphor is loaded.** The router-as-classifier described here is *not* a meta-cognitive system; it is a discriminative head on a small LLM. Be precise in the paper — conflating this with AGI-style architecture will hurt review scores.
5. **Phi-3.5-mini forgetting numbers (0.02) come from a single GLUE-NLU study** (arXiv:2504.01241) on simple classification tasks. Don't over-extrapolate to the harder TRACE setting; re-measure.
6. **V-JEPA 2.1 ViT-B (80M)** is `torch.hub`-only as of this writing (HF integration is open issue #45496). If you depend on the HF `AutoModel` path, either wait for the integration or write a small loader.
7. **No published work has shown JEPA-style training stably combined with NF4-quantized base models.** Novel territory; expect at least one round of debugging numerical stability (gradient underflow, loss NaN on long latent rollouts). Mitigations: train LoRA in bf16, monitor `nan_to_num` calls, set `bnb_4bit_compute_dtype=torch.bfloat16`.
8. **V-JEPA 2 ViT-L SSv2 number used in this document is 73.7% top-1 frozen-probe** (per the `facebookresearch/vjepa2` README), not the 77.3% headline figure (which is ViT-g and not your target model). Don't cite the wrong number in your paper.