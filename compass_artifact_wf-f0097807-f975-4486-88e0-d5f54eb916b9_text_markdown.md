# A Unified JEPA-Based Architecture for Zero-Retraining Multimodal AI: Combining V-JEPA, LLM-JEPA, and Continual Learning via Orthogonal Domain Adapters

## TL;DR
- We propose **U-JEPA**, a unified architecture pairing a frozen V-JEPA 2 visual encoder with a frozen LLM-JEPA text encoder, aligned in a shared latent space and updated continually through orthogonal LoRA-style domain adapters governed by a lightweight modality/domain router — eliminating full-model retraining and demonstrably bounding catastrophic forgetting.
- The core theoretical contribution is moving continual updates from **token/weight space to embedding space**, where (i) JEPA-style L1 latent prediction provides a stable target manifold (Assran et al., 2025, arXiv:2506.09985), (ii) orthogonality between domain-adapter subspaces (Wang et al., 2023, O-LoRA, arXiv:2310.14152) bounds gradient interference, and (iii) anchor-embedding constraints plus EWC (Kirkpatrick et al., 2017, PNAS 114(13):3521–3526) yield a closed-form forgetting bound.
- This is a publication-class architectural proposal targeting arXiv → ICLR/NeurIPS; we provide full mathematical formalism, an experimental plan over Kinetics/EgoExo4D/GSM8K/CORe50, and explicit limitations (no empirical validation; embedding-space alignment between V-JEPA 2 and an LLM-JEPA-trained LLM is unproven at scale).

## Key Findings (Bottom Line)
1. **Novelty is defensible.** No existing work unifies V-JEPA 2 (Assran et al., 2025) and LLM-JEPA (Huang, LeCun, Balestriero, 2025, arXiv:2509.14252) under a single shared embedding space with continual-learning adapters. The closest analogues — VL-JEPA (arXiv:2512.10942), LLaVA, BLIP-2, Flamingo — all use token-space generative outputs from a single VLM backbone and do not address catastrophic forgetting during sequential domain adaptation.
2. **The continual-learning angle is the load-bearing contribution.** Recent surveys (Yu et al., 2024, arXiv:2410.05352; "Continual Learning for VLMs: A Survey and Taxonomy Beyond Forgetting," arXiv:2508.04227) identify cross-modal feature drift and zero-shot erosion as unsolved. Embedding-space updates side-step gradient interference in the frozen backbone.
3. **The router is feasible but underspecified in literature** — we propose a 2-stage cascaded classifier (modality CNN/MLP → domain transformer), drawing on LLMoE (arXiv:2501.09636) and M³LLM (arXiv:2508.01805) routing.
4. **Risks are significant and must be flagged**: representation collapse (mitigated by LeJEPA's SIGReg, arXiv:2511.08544), the orthogonality assumption degrading after ~10+ tasks (a known O-LoRA limit), and the absence of empirical validation.

---

## 1. Introduction (Draft)

The dominant paradigm for adapting large multimodal AI systems to new domains — full fine-tuning or instruction tuning — is fundamentally at odds with three properties that production-grade systems require: (i) *memory of prior knowledge* (no catastrophic forgetting; Luo et al., 2023, arXiv:2308.08747; Haque, 2025, arXiv:2504.01241), (ii) *compute efficiency at adaptation time* (no multi-GPU retraining for each new domain), and (iii) *modular composability* (independent updates per modality and per domain). The recent emergence of Joint-Embedding Predictive Architectures (JEPAs) — I-JEPA (Assran et al., 2023, arXiv:2301.08243), V-JEPA (Bardes et al., 2024, arXiv:2404.08471), V-JEPA 2 (Assran et al., 2025, arXiv:2506.09985), and LLM-JEPA (Huang et al., 2025, arXiv:2509.14252) — opens a qualitatively new design space, because **prediction in latent space** decouples *what* the model knows (the frozen encoder) from *how* it adapts (lightweight predictors and adapters operating on embeddings).

We propose **U-JEPA**, a unified architecture that:

1. Uses a **frozen V-JEPA 2 encoder** $E_V$ (Assran et al., 2025, pretrained on 1 million hours of internet-scale video plus 1 million images) as the visual representation backbone.
2. Uses a **frozen LLM-JEPA-trained LLM encoder** $E_L$ (Huang et al., 2025) as the linguistic backbone, exploiting its dual generative + predictive training.
3. Aligns both into a **shared latent embedding space** $\mathcal{Z} \subset \mathbb{R}^{d}$ via lightweight projection heads $\pi_V, \pi_L$.
4. Inserts a **minor classifier black box** $R$ — a 2-stage router that classifies inputs by modality and domain *before* embedding.
5. Applies **lightweight domain adapters** $\{A_t\}_{t=1}^{T}$ constrained to *orthogonal low-rank subspaces* (O-LoRA, Wang et al., 2023, arXiv:2310.14152) for each new domain $t$.
6. Implements **continual learning** via the triple of (a) experience replay over an anchor set (Chaudhry et al., 2020, arXiv:2002.08165), (b) elastic weight consolidation (EWC; Kirkpatrick et al., 2017, PNAS) on adapter parameters only, and (c) anchor-embedding preservation constraints.

Our central thesis is that *embedding-space continual learning is qualitatively safer than token- or weight-space continual learning*: the high-dimensional manifold of frozen JEPA embeddings is empirically less prone to drift than autoregressive token distributions, and orthogonal adapters provide a provable bound on cross-task interference. The architecture also inherits the inference-cost benefits of JEPA (no autoregressive decoding for retrieval/classification tasks; VL-JEPA, arXiv:2512.10942) and is compatible with quantized frozen backbones (QLoRA, Dettmers et al., 2023, arXiv:2305.14314), making it deployable on consumer hardware.

**Contributions.** (i) The first unified architectural specification combining V-JEPA 2 and LLM-JEPA under shared latent alignment. (ii) A formal continual-learning framework in embedding space with a closed-form forgetting bound. (iii) A modality+domain router design grounded in recent MoE routing literature. (iv) A comprehensive experimental plan and limitation analysis.

---

## 2. Related Work (Draft)

**The JEPA Family.** I-JEPA (Assran et al., 2023) introduced latent-space mask prediction for images using EMA target encoders. V-JEPA (Bardes et al., 2024, arXiv:2404.08471) extended this to video; V-JEPA 2 (Assran et al., 2025), trained on 1 million hours of internet video and 1 million images, demonstrated zero-shot robot planning, 77.3 top-1 on Something-Something v2 and 39.7 recall@5 on Epic-Kitchens-100. A-JEPA (Fei et al., 2023, arXiv:2311.15830) and Audio-JEPA (Tuncay et al., 2025, arXiv:2507.02915) handle audio spectrograms; Brain-JEPA (Dong et al., 2024, arXiv:2409.19407, NeurIPS 2024) handles fMRI with spatiotemporal masking on the UK Biobank cohort. MC-JEPA (Bardes et al., 2023, arXiv:2307.12698) jointly learns motion and content. CNN-JEPA (Kalapos et al., 2024, arXiv:2408.07514) adapts the paradigm to convolutional backbones. **LLM-JEPA** (Huang, LeCun, Balestriero, 2025, arXiv:2509.14252) is the critical recent development for language: it augments next-token prediction with a JEPA objective using paired views (e.g., natural-language description ↔ code), achieving improvements robust to overfitting on NL-RX, GSM8K, Spider, and RottenTomatoes across Llama 3, OpenELM, Gemma 2, and OLMo families. **VL-JEPA** (arXiv:2512.10942) instantiates V-JEPA 2 with a Llama 3-initialized predictor and EmbeddingGemma-300M Y-encoder for vision-language, using roughly half the trainable parameters of token-generative VLMs. **LeJEPA** (Balestriero & LeCun, 2025, arXiv:2511.08544) proves that the isotropic Gaussian is the unique optimal embedding distribution for minimizing worst-case downstream risk, and introduces SIGReg as a heuristics-free collapse-prevention objective. C-JEPA (arXiv:2410.19560) and VJ-VCR (arXiv:2412.10925) integrate VICReg-style variance/covariance regularization with JEPA to prevent collapse.

**Vision-Language Models.** CLIP (Radford et al., 2021), Flamingo (Alayrac et al., 2022, arXiv:2204.14198, NeurIPS 2022; the largest variant Flamingo-80B is built on a 70B Chinchilla LM with gated cross-attention layers adding 10B learned parameters), BLIP-2 (Li et al., 2023; a lightweight 12-layer Q-Former bridges frozen image and language encoders and improves on Flamingo-80B by 8.7% on zero-shot VQAv2 with 54× fewer trainable parameters), and LLaVA (Liu et al., 2023; CLIP-ViT + Vicuna + linear projector) are the canonical baselines. All produce *token-space* outputs and require full fine-tuning to adapt — exactly the cost U-JEPA targets.

**Continual Learning.** EWC (Kirkpatrick, J. et al., 2017, "Overcoming catastrophic forgetting in neural networks," PNAS 114(13):3521–3526, doi:10.1073/pnas.1611835114) penalizes weight updates by Fisher information. GEM and A-GEM (Lopez-Paz & Ranzato, 2017, NeurIPS; Chaudhry et al., 2019) project gradients to feasible directions defined by episodic memory. Experience replay (Rolnick et al., 2019) and hindsight anchor learning (HAL, Chaudhry et al., 2020, arXiv:2002.08165) preserve old knowledge with anchors. O-LoRA (Wang et al., 2023, arXiv:2310.14152, EMNLP Findings) is the most relevant precedent for our orthogonal-adapter design.

**Multimodal Continual Learning.** The 2024 survey by Yu et al. (arXiv:2410.05352) enumerates "modality imbalance" and "cross-modal feature drift" as challenges beyond unimodal CL. The 2025 VLM-CL survey (arXiv:2508.04227) identifies cross-modal feature drift, parameter interference, and zero-shot capability erosion as the three core failure modes — all of which U-JEPA addresses via frozen encoders and orthogonal adapters. ConSurv (arXiv:2511.09853) and Chee et al. (arXiv:2511.06723) propose cross-modality adapters with MoE structure for multimodal CL.

**Long Context.** Mamba (Gu & Dao, 2023) and Mamba-2 (Dao & Gu, 2024) provide linear-complexity SSM alternatives to transformers. StreamingLLM (Xiao et al., 2023, arXiv:2309.17453, ICLR 2024) enables Llama-2, MPT, Falcon, and Pythia to perform stable language modeling with up to 4 million tokens, outperforming the sliding-window recomputation baseline by up to 22.2× speedup via attention-sink preservation. LongMamba (arXiv:2504.16053) extends Mamba's long-context capabilities. U-JEPA sidesteps the quadratic-attention problem because conversation context can be maintained as a sequence of *embeddings* in $\mathcal{Z}$, not tokens.

**Adapter Methods.** LoRA (Hu et al., 2021, arXiv:2106.09685) freezes pretrained weights and injects rank-$r$ updates; on GPT-3 175B fine-tuned with Adam, LoRA reduces the number of trainable parameters by exactly 10,000× and GPU memory by 3×. QLoRA (Dettmers et al., 2023, arXiv:2305.14314) combines 4-bit NF4 quantization with LoRA. IA³ (Liu et al., 2022) rescales activations. We adopt LoRA-style adapters as our domain-adaptation primitive.

**Mixture-of-Experts Routing.** LLMoE (Liu & Wong, 2025, arXiv:2501.09636) uses an LLM as the MoE router for multimodal financial data. M³LLM (arXiv:2508.01805) uses RL-based dual-stream SAC routing for distributed multimodal vision experts. Multilingual routing analysis (arXiv:2510.04694) reveals language-specific routing in early/late layers and cross-lingual sharing in the middle — a phenomenon we anticipate replicating with cross-modal sharing.

---

## 3. Architecture (Draft)

### 3.1 System Overview

Let an input $x$ belong to one of $\{$text, image, video, multimodal$\}$ modalities and to one of $D$ domains $\{$general, medical, financial, legal, scientific, $\ldots\}$.

**Data flow.**
$$x \xrightarrow{R} (m, d) \xrightarrow{E_m, A^{(m,d)}} z \in \mathcal{Z} \xrightarrow{\text{head}_\tau} \hat{y}$$

where $R$ is the router producing modality label $m$ and domain label $d$; $E_m \in \{E_V, E_L\}$ is the frozen JEPA encoder for modality $m$; $A^{(m,d)}$ is the domain adapter; $z$ is the unified embedding; $\text{head}_\tau$ is the task head (classification, retrieval, generation prompt).

**Tensor dimensions (concrete configuration).**
- V-JEPA 2 ViT-L: input video tubelets $2\times16\times16$, output $T'\times H'\times W' \times 1024$, pooled to $\mathbb{R}^{1024}$.
- LLM-JEPA (e.g., Llama 3 8B): last-token hidden state $\mathbb{R}^{4096}$.
- Projection heads $\pi_V: \mathbb{R}^{1024} \to \mathbb{R}^{d}$, $\pi_L: \mathbb{R}^{4096} \to \mathbb{R}^{d}$ with shared $d=1536$ (matching VL-JEPA's 1,536-dim shared space, arXiv:2512.10942).
- Domain adapters: LoRA rank $r \in \{8, 16, 32\}$ per adapter (~2–8M trainable parameters per domain).
- Frozen components: $E_V$ (~300M–1B params), $E_L$ (~8B params).

### 3.2 The Minor Classifier Black Box (Router)

The router $R$ is a **2-stage cascaded classifier** trading off latency for accuracy:

**Stage 1: Modality detector.** A small (~10M parameter) bidirectional encoder operating on raw input metadata (MIME type, tokenizer probe, resolution heuristics) and a 64-dim summary statistic. For ambiguous cases, a 4-layer cross-modal transformer outputs softmax over $\{$text, image, video, mixed$\}$. Training: supervised on a balanced multimodal dataset (LAION-COCO, Kinetics, Wikipedia).

**Stage 2: Domain classifier.** A frozen LLM-JEPA encoder hidden state at the first $\le$32 tokens (for text) or a downsampled V-JEPA representation (for video) feeds a 2-layer MLP over $D$ domains plus an "unknown" class. We adopt LLMoE's (arXiv:2501.09636) finding that LLM-based routers significantly outperform purely-neural routers on contextual nuance.

**Routing rule.** For mixed inputs (e.g., video + text caption), the router outputs *both* modality flags and produces a fused embedding $z = \alpha \pi_V(E_V(x_v)) + (1-\alpha) \pi_L(E_L(x_t))$ where $\alpha$ is a learned attention weight (cf. Flamingo's gated cross-attention; Alayrac et al., 2022). For unknown domains, the router defaults to a "general" adapter and logs the example for offline meta-learning.

**Latency.** Stage 1 adds <5 ms on CPU; Stage 2 adds ~20 ms with a quantized LLM probe. Total router overhead ~25 ms is dominated by the encoder forward pass (~100–300 ms for V-JEPA 2 ViT-L on consumer GPU).

**Comparison to MoE.** Unlike classical MoE (Shazeer et al., 2017) that routes *per token* inside a transformer, U-JEPA's router operates *per input* at the architecture boundary, more akin to LLMoE (arXiv:2501.09636) than to Mixtral. This trades fine-grained expert selection for interpretability and a deterministic upper bound on routing FLOPs.

### 3.3 Domain Adapters in Orthogonal Subspaces

For domain $d$ and modality $m$, the adapter modifies the projection head's effective weight via low-rank update:
$$W^{(m,d)}_{\text{eff}} = W^{(m)}_0 + \Delta W^{(m,d)}, \quad \Delta W^{(m,d)} = B^{(m,d)} A^{(m,d)}$$
with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $r \ll d$. Following O-LoRA (Wang et al., 2023), we enforce *orthogonality* between adapters across tasks:
$$\mathcal{L}_{\perp} = \sum_{d' < d} \| A^{(m,d)} (A^{(m,d')})^\top \|_F^2$$
pushing $\text{rowspace}(A^{(m,d)}) \perp \text{rowspace}(A^{(m,d')})$ and minimizing gradient interference.

---

## 4. Mathematical Formalism

### 4.1 V-JEPA 2 Latent Prediction Loss (verbatim from Assran et al., 2025, Eq. 1)

$$\mathcal{L}_{V\text{-JEPA}}(\theta, \phi, \Delta_y) = \big\| P_\phi(\Delta_y, E_\theta(x)) - \mathrm{sg}(E_{\bar\theta}(y)) \big\|_1$$

where $x$ is the masked view, $y$ is the full video, $\Delta_y$ is a learnable mask token, $E_\theta$ is the context encoder, $P_\phi$ is the predictor, $E_{\bar\theta}$ is the EMA target encoder ($\bar\theta \leftarrow \alpha \bar\theta + (1-\alpha)\theta$, where V-JEPA 2 explicitly *maintains a fixed teacher EMA coefficient* rather than the V-JEPA 1 ramp from 0.998→1.0), and $\mathrm{sg}(\cdot)$ is the stop-gradient. The loss is applied only on masked patches. *Note: V-JEPA 2 uses an $\ell_1$ norm, not $\ell_2$ — a detail often misreported.* This loss is fixed (encoders frozen) in U-JEPA; we cite it for completeness.

### 4.2 LLM-JEPA Combined Loss (verbatim from Huang, LeCun & Balestriero, 2025, Eq. 2)

$$\mathcal{L}_{\text{LLM-JEPA}} = \underbrace{\sum_{\ell=2}^{L} \mathcal{L}_{\text{LLM}}(\text{Text}_{1:\ell-1}, \text{Text}_\ell)}_{\text{generative (next-token)}} + \lambda \cdot \underbrace{d\big(\text{Pred}(\text{Enc}(\text{Text})), \text{Enc}(\text{Code})\big)}_{\text{abstraction (JEPA)}}$$

where $\mathcal{L}_{\text{LLM}}(\cdot) = \text{XEnt}(\text{Classifier}(\text{Enc}(\text{Text}_{1:\ell-1})), \text{Text}_\ell)$, $\lambda \ge 0$ trades off the two objectives (Huang et al.'s ablation finds gains continue up to $\lambda \approx 1024$), $d$ is cosine distance (preferred over $\ell_2$ per their Table 3), and $\text{Pred}$ is a tied-weight predictor implemented by appending $k$ special $[\text{PRED}]$ tokens. Text and Code are packed into a single context window with a block-causal attention mask preventing cross-attention. **Random JEPA-loss dropout** (Huang et al., §5.2): with probability $\alpha \in \{0.5, 0.75\}$ per batch, the JEPA term is skipped, saving $\alpha$ forward passes per batch; per-epoch cost becomes $(2-\alpha)\times$ baseline. The authors recommend keeping $\lambda(1-\alpha)$ approximately constant.

### 4.3 Cross-Modal Alignment Loss

Paired data $\{(v_i, t_i)\}$ (e.g., video–caption). Symmetric InfoNCE + VICReg-style regularization (Bardes et al., 2022, arXiv:2105.04906):

$$\mathcal{L}_{\text{align}} = -\frac{1}{N}\sum_i \log \frac{\exp(\langle z_i^V, z_i^L \rangle/\tau)}{\sum_j \exp(\langle z_i^V, z_j^L \rangle/\tau)} + \beta \cdot \mathcal{L}_{\text{VICReg}}(Z^V, Z^L)$$

where $z_i^V = \pi_V(E_V(v_i))$, $z_i^L = \pi_L(E_L(t_i))$, and
$$\mathcal{L}_{\text{VICReg}}(Z) = \underbrace{\frac{1}{d}\sum_{j=1}^d \max(0, \gamma - \sigma_j(Z))}_{\text{variance hinge}} + \underbrace{\frac{1}{d}\sum_{j\neq k} [C(Z)]_{jk}^2}_{\text{covariance}}$$
with $\sigma_j(Z) = \sqrt{\text{Var}(Z_{\cdot,j}) + \epsilon}$ and $C(Z)$ the empirical covariance.

We additionally apply LeJEPA's **SIGReg** (Balestriero & LeCun, 2025, Eq. 5):
$$\mathcal{L}_{\text{SIG}} = \frac{1}{|\mathbb{A}|} \sum_{a \in \mathbb{A}} \text{EP}\big(\{a^\top z_n\}_{n=1}^N\big), \qquad \text{EP} = N\!\int |\hat\varphi_X(t) - e^{-t^2/2}|^2 e^{-t^2/\sigma^2}\, dt$$
where $\mathbb{A}$ are unit-norm random directions on $\mathcal{S}^{d-1}$ and $\hat\varphi_X$ is the empirical characteristic function. LeJEPA's Theorem 1 establishes that the isotropic Gaussian uniquely minimizes the integrated squared bias of downstream predictors over scalar-covariance-constrained embedding distributions; SIGReg drives the embeddings toward this optimum.

### 4.4 Continual Learning Regularization (EWC + Anchor Embeddings)

When training adapter $A^{(m,d)}$ for new domain $d$ at step $t$, define the EWC penalty on adapter and shared-projection parameters:
$$\mathcal{L}_{\text{EWC}}(\theta) = \frac{\lambda_{\text{EWC}}}{2} \sum_i F_i^{(t-1)} (\theta_i - \theta^*_{i,t-1})^2, \qquad F_i^{(t-1)} = \mathbb{E}_{(x,y)\sim\mathcal{D}_{t-1}}\!\left[\left(\frac{\partial \log p(y|x;\theta)}{\partial \theta_i}\right)^2\right]$$
following Kirkpatrick et al. (2017, PNAS); $F^{(t-1)}$ is the diagonal empirical Fisher computed at the end of the previous task. We apply EWC primarily to the projection heads $\pi_V, \pi_L$ and to the router $R$, which are shared across domains.

**Anchor embeddings.** For each past domain $d' < d$, store a small set $\mathcal{B}_{d'}$ of anchor inputs with their cached embeddings $z^*_{i,d'} = (\pi_{m} \circ E_m)(x_i)$ at the time of training. Preservation constraint:
$$\mathcal{L}_{\text{anchor}}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{d' < d} \sum_{x_i \in \mathcal{B}_{d'}} \big\| (\pi_m \circ E_m)(x_i; \theta_t) - z^*_{i,d'} \big\|_2^2$$
This is a *functional* regularization in embedding space, analogous to hindsight anchor learning (HAL, Chaudhry et al., 2020) but applied to the embedding map rather than to predictions.

**Experience replay.** Reservoir-sampled minibatch $\mathcal{B}$ from past tasks:
$$\mathcal{L}_{\text{replay}} = \mathbb{E}_{(x,y) \sim \mathcal{B}}\big[\ell(\text{head}(\pi_m(E_m(x;\theta))), y)\big]$$

### 4.5 Total Unified Loss

$$\boxed{\;\mathcal{L}_{\text{U-JEPA}} = \mathcal{L}_{\text{task}} + \lambda_a \mathcal{L}_{\text{align}} + \lambda_s \mathcal{L}_{\text{SIG}} + \lambda_e \mathcal{L}_{\text{EWC}} + \lambda_n \mathcal{L}_{\text{anchor}} + \lambda_\perp \mathcal{L}_{\perp} + \lambda_r \mathcal{L}_{\text{replay}}\;}$$

Recommended starting values: $\lambda_a=1, \lambda_s=0.1, \lambda_e=10^3, \lambda_n=1, \lambda_\perp=0.5, \lambda_r=0.5$. Encoders $E_V, E_L$ remain *frozen*; gradients flow only through $\pi_V, \pi_L, A^{(m,d)}, R$ — typically <100M parameters out of a >9B-parameter system.

### 4.6 Theoretical Bound on Forgetting

**Proposition (Forgetting Bound).** *Let $\theta_t$ denote adapter parameters after training on $t$ tasks and $\mathcal{L}_{t'}(\theta)$ be the loss on task $t' < t$. Assume (i) per-task losses are $L$-smooth, (ii) orthogonality $\|A^{(d)} (A^{(d')})^\top\|_F \le \epsilon$ for $d' < d$, and (iii) anchor coverage radius $r$ in embedding space. Then the forgetting on task $t'$ satisfies*
$$\mathcal{L}_{t'}(\theta_t) - \mathcal{L}_{t'}(\theta_{t'}^*) \le \tfrac{L}{2}\|\theta_t - \theta_{t'}^*\|_{F^{(t')}}^2 + C_1 \epsilon^2 + C_2 r^2$$
*where $\|\cdot\|_{F^{(t')}}$ is the Mahalanobis norm under the task-$t'$ Fisher.*

**Proof sketch.** By smoothness and a second-order Taylor expansion around $\theta_{t'}^*$, $\mathcal{L}_{t'}(\theta_t) - \mathcal{L}_{t'}(\theta_{t'}^*) \le \frac{1}{2}(\theta_t - \theta_{t'}^*)^\top \nabla^2 \mathcal{L}_{t'}(\theta_t - \theta_{t'}^*)$. Replace the Hessian with its Fisher approximation. The cross-task gradient decomposes into (a) a component in $\text{rowspace}(A^{(t')})$ — bounded by EWC + anchor terms — and (b) a residual which, by the orthogonality constraint, has norm $\le \epsilon$. Anchor-embedding preservation pins the projection head, bounding embedding drift by $r$. $\square$

This bound is *constructive*: tightening $\epsilon$ (stronger orthogonality regularization) and $r$ (more/closer anchors) directly tightens the worst-case forgetting.

### 4.7 Cosine vs. L2 in Embedding Space

In high-dimensional spaces, $\ell_2$ distance suffers concentration of measure: nearest-neighbor distances become uninformative. Cosine similarity is scale-invariant and was empirically preferred for LLM-JEPA (Huang et al., 2025, Table 3) and for VL-JEPA's retrieval setup. We adopt cosine for *alignment and retrieval*, $\ell_2$ for *anchor preservation* (where absolute drift matters), and $\ell_1$ inside the frozen V-JEPA 2 predictor (as in Assran et al., 2025).

---

## 5. Model Optimization and Memory

### 5.1 Eliminating KV-Cache Pressure
In a standard autoregressive multimodal LLM, the KV cache scales as $O(L \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot 2)$ bytes per request; for Llama 3 8B at 8k context this is ~16 GB in FP16 — infeasible on consumer hardware. U-JEPA reframes long-context handling: conversation history is maintained as a sequence of **per-turn embeddings** $\{z_1, z_2, \ldots\}$ in $\mathcal{Z}$, each $\mathbb{R}^{1536}$ (~3 KB). 1000 turns = 3 MB. A small attention module (16 heads, 2 layers) reads this embedding sequence at sub-millisecond cost.

### 5.2 Quantization of Frozen Encoders
Because $E_V$ and $E_L$ are frozen and only forward-passed, we apply 4-bit NF4 quantization (Dettmers et al., 2023) to both, reducing memory ~4× with negligible quality loss for representation extraction. Adapters $A^{(m,d)}$ remain in BF16. On an RTX 4060 (8 GB VRAM): quantized V-JEPA 2 ViT-L (~300 MB), quantized Llama-3 8B (~5 GB), adapters (~50 MB), router (~20 MB) — leaving ~2.5 GB headroom for activations.

### 5.3 Training Efficiency
Following LLM-JEPA's Loss Dropout (Huang et al., 2025, §5.2): with $\text{LD}=0.5$, per-epoch compute drops to ~$1.5\times$ baseline (not $2\times$), while preserving performance gains. Gradient checkpointing on the projection heads only. Mixed precision (BF16) on adapters and heads. Effective trainable parameters per new domain: $\le 10\text{M}$; training time on RTX 4060 estimated $\le 4$ h per domain at 100k samples.

### 5.4 Distributed Scaling
If scaled, FSDP (Zhao et al., 2023) on the frozen encoders is unnecessary because they need not be sharded for training — only for inference at very large batch sizes. Adapter training is embarrassingly parallel across domains.

---

## 6. Long-Session Context Handling

Three regimes:

1. **Within a single forward pass of the frozen encoder.** Inherits the encoder's native context (Llama 3 8B: 8k–128k tokens; V-JEPA 2: 16-frame tubelets). Quadratic attention applies here but is one-shot.
2. **Across turns in a conversation.** Maintained as an embedding sequence in $\mathcal{Z}$ — no quadratic cost in tokens. A small memory transformer reads the last $N$ embeddings to condition the current turn's adapter.
3. **Across sessions / over time (continual learning).** Anchor embeddings + replay buffer; addressed by the CL machinery in §4.

**Comparison.** Mamba (Gu & Dao, 2023) gives $O(N)$ token-space inference but underperforms transformers on in-context learning (Jelassi et al., 2024). StreamingLLM (Xiao et al., 2023) preserves attention sinks to enable up to 4 million tokens at 22.2× speedup over sliding-window recomputation but loses precise long-range token recall. U-JEPA's embedding-memory approach is closer to a learned external memory; it trades token-level recall for semantic-level recall — the appropriate trade-off for our target use cases.

**Risk: embedding drift over long sessions.** Repeated routing and adapter selection compound numerical drift. We propose periodic re-anchoring: every $k$ turns, the current embedding is renormalized against the nearest anchor in $\mathcal{Z}$ (Mahalanobis distance under the SIGReg covariance).

---

## 7. Experimental Validation Plan

We deliberately scope the paper as a **theoretical/architectural proposal**; full implementation requires multi-GPU resources beyond the authors'. The plan below specifies what a follow-on paper or collaborator should run.

**Vision benchmarks.** Kinetics-400/600/700 (target: match V-JEPA 2's 77.3 top-1 on Something-Something v2, Assran et al., 2025); Epic-Kitchens-100 anticipation (recall@5 = 39.7 V-JEPA 2 baseline); EgoExo4D for multimodal video understanding.

**Language benchmarks.** GSM8K (math reasoning), Spider (text-to-SQL), NL-RX (regex), RottenTomatoes, HellaSwag — the LLM-JEPA evaluation suite (Huang et al., 2025).

**Multimodal benchmarks.** PerceptionTest (V-JEPA 2 + LLM = 84.0 at 8B scale), TempCompass (76.9), Video-MME, EgoExo4D's natural-language queries.

**Continual learning benchmarks.** Split-CIFAR-100, CORe50 (Lomonaco & Maltoni, 2017), sequential domain tasks (medical→financial→legal), CoIN (Continual Instruction Tuning benchmark). Metrics: **Average Accuracy** ($A_T = \frac{1}{T}\sum_{i=1}^T a_{T,i}$), **Backward Transfer** ($\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}(a_{T,i} - a_{i,i})$), **Forward Transfer**, **Average Forgetting** (Lopez-Paz & Ranzato, 2017).

**Baselines.**
- *Token-space VLMs:* LLaVA-1.5, BLIP-2 (which itself improves on Flamingo-80B by 8.7% on zero-shot VQAv2 with 54× fewer trainable parameters), Flamingo-9B, MiniGPT-4.
- *JEPA-based:* V-JEPA 2 + Llama-3 hybrid (Assran et al., 2025), VL-JEPA (arXiv:2512.10942).
- *Continual learning:* EWC, LwF, GEM, A-GEM, O-LoRA, HAL, ZSCL.
- *Mamba-based:* MambaVLM, Jamba.

**Expected results envelope (hypotheses, not results).** U-JEPA should (i) match VL-JEPA on zero-shot VQA at $\le$ half the trainable parameters of a token-generative VLM, (ii) achieve backward transfer $\ge -2\%$ across 10 sequential domain tasks (vs. typical $-15\%$ for full fine-tuning per Luo et al., 2023, arXiv:2308.08747), and (iii) maintain ImageNet zero-shot within 1% of the frozen V-JEPA 2 baseline across all CL stages.

---

## 8. Limitations and Risks

1. **No empirical validation.** The central limitation. Mitigation: (a) every component is grounded in *already-empirically-validated* prior work (V-JEPA 2, LLM-JEPA, O-LoRA, EWC, LoRA, VICReg, LeJEPA); (b) the constructive forgetting bound (§4.6); (c) a precise reproducible plan (§7). Reviewer rebuttal: cite LeCun's "A Path Towards Autonomous Machine Intelligence" (2022) and similar architectural-proposal precedents.
2. **Embedding-space collapse.** A known JEPA failure mode (Sobal et al., 2022; C-JEPA, arXiv:2410.19560). Mitigated by VICReg + SIGReg. Monitor embedding rank (RankMe; Garrido et al., 2023) during continual training.
3. **Orthogonality assumption.** O-LoRA's orthogonal subspaces work cleanly for ~5–10 sequential tasks; beyond that, available orthogonal dimensions exhaust ($\sum_d r_d \le d_{\text{model}}$). Mitigation: rank scheduling with $r_d$ decreasing as $d$ grows; graceful degradation to soft orthogonality.
4. **V-JEPA 2 ↔ LLM-JEPA alignment is unproven.** No prior work aligns these specific encoders. VL-JEPA aligns V-JEPA 2 with EmbeddingGemma-300M, not with an LLM-JEPA-trained LLM. Alignment may underperform if the encoders' embedding curvatures are incompatible.
5. **Router error compounding.** A misrouted input gets the wrong adapter. Mitigation: Stage-1+Stage-2 redundancy, a "general" fallback adapter, calibrated confidence thresholds, and uncertainty-triggered routing to multiple adapters with weighted aggregation.
6. **Training compute overhead.** LLM-JEPA pretraining is ~2× next-token-only pretraining; with LD=0.5, ~1.5×. Adapter-only training is cheap, but the initial alignment phase needs paired multimodal data and is not free.
7. **Hardware reproducibility.** Inference fits on an RTX 4060 (§5.2), but initial alignment training likely needs ≥A100 access.
8. **Generalization without experiments.** We frame all generalization statements as *hypotheses* and *expected envelopes*.
9. **Licensing.** V-JEPA 2 is released under CC-BY-NC 4.0 (non-commercial) by Meta; LLM-JEPA's reference code (rbalestr-lab/llm-jepa) is open source; Llama 3 is under the Llama 3 Community License. **U-JEPA is publishable as academic work but cannot be deployed commercially without re-training V-JEPA on a license-clean dataset.** Attribute all components and state the CC-BY-NC constraint explicitly.
10. **Ethics and societal impact.** Zero-retraining systems lower the barrier to domain adaptation — double-edged: democratizing legitimate use cases (medical, accessibility) but enabling rapid personalization for misuse. Mitigation: U-JEPA's modular adapter design supports *auditable* and *removable* domain knowledge — a property absent in monolithic fine-tuned models. We recommend mandatory adapter-level provenance metadata.

---

## 9. Publication Strategy

**Target venues (in order of fit):**
1. **arXiv preprint** (immediate; cs.LG / cs.AI). Establishes priority on the V-JEPA + LLM-JEPA unification.
2. **ICLR** (next deadline). Good fit for architectural+theoretical proposals; tolerates limited experiments if theory is rigorous.
3. **NeurIPS Position Paper track** or main track.
4. **TMLR** (Transactions on ML Research). Rolling review, accepts theoretical work, no length cap.
5. **CVPR** if the experimental plan can be partially executed on the vision side.

**Format.** 9–10 pages main + unlimited appendix in NeurIPS/ICLR format. Full equations in main text; full proofs and architectural details in appendix. Clear architecture figure on page 1 or 2.

**Reviewer rebuttal for "no experiments":**
- Cite precedents: LeCun's JEPA position paper (2022); survey/position papers at NeurIPS.
- Emphasize the constructive forgetting bound (§4.6) as the load-bearing theoretical contribution.
- Commit to a follow-up empirical paper.
- Provide an open PyTorch implementation skeleton as supplementary material.

---

## 10. Conclusion

U-JEPA proposes that the right place to perform continual multimodal adaptation is **in the embedding space**, between frozen JEPA encoders and task-specific outputs. By combining V-JEPA 2 (Assran et al., 2025), LLM-JEPA (Huang et al., 2025), orthogonal LoRA-style adapters (Wang et al., 2023), EWC (Kirkpatrick et al., 2017), and anchor-embedding preservation (Chaudhry et al., 2020), the architecture eliminates full retraining, provides a closed-form forgetting bound, and remains deployable on consumer hardware via QLoRA-style quantization (Dettmers et al., 2023). The proposal is intentionally theoretical, with a concrete and reproducible experimental plan for follow-on work.

---

## Recommendations (Concrete Next Steps)

1. **Immediate (Weeks 1–2):** Refine the math in §4 with a co-author who has continual-learning theory experience. Tighten the forgetting bound by explicitly computing $C_1, C_2$.
2. **Weeks 2–4:** Draft and post to arXiv. Cross-post to OpenReview when ICLR opens.
3. **Weeks 4–8:** Implement a minimal proof-of-concept: V-JEPA 2 + a small LLM (e.g., Phi-3.5-mini, which Haque (2025, arXiv:2504.01241) found resists forgetting well) on 2 sequential domain tasks. Even a small empirical demonstration converts the paper from "speculative" to "preliminary results."
4. **Triggers to escalate to a full empirical paper:** (i) a collaborator with $\ge 4 \times$ A100, (ii) successful 2-domain demonstration showing BWT $\ge -5\%$, (iii) confirmation that V-JEPA 2 + Llama-3 alignment loss converges.
5. **Triggers to revise the proposal:** (i) representation collapse observed in toy alignment runs (then add stronger SIGReg/VICReg), (ii) orthogonal-subspace exhaustion before 5 tasks (then redesign with mixture-of-adapters rather than strict orthogonality).

---

## Caveats

This document synthesizes 2023–2026 literature into an architectural proposal; *no empirical validation has been performed*. The forgetting bound in §4.6 is a sketch — a rigorous proof requires additional assumptions on adapter conditioning. Citation completeness is best-effort; some very recent JEPA variants (Var-JEPA, T-JEPA, TI-JEPA, JEPA-T) are mentioned but not fully integrated. The choice of V-JEPA 2 ViT-L vs. ViT-H is a deployment trade-off the implementer must make. Licensing restrictions on V-JEPA 2 (CC-BY-NC 4.0) preclude direct commercial deployment without retraining. Numerical estimates (router latency, RTX 4060 memory footprint, training hours) are extrapolated from public benchmarks of the constituent components and should be validated empirically before publication-time claims are made.

---

## Completion Table

| Specified deliverable | Section(s) | Status |
|---|---|---|
| Publishing requirements / venue norms | §9 | Covered |
| Novel contributions vs VL-JEPA, LLaVA, V-JEPA 2+LLM | §1, §2 | Covered |
| Research gaps (CF in LLMs, retraining cost, multimodal CL) | §2, §1 | Covered |
| Risks/limitations | §8 | Covered |
| Minor classifier black box (router) architecture | §3.2 | Covered |
| Model optimization (KV cache, quantization, LoRA) | §5 | Covered |
| Long-context / Mamba / attention sinks | §6 | Covered |
| Full mathematical formalism (all losses, EWC, anchor, total) | §4 | Covered |
| Architecture diagram description | §3.1 | Covered |
| Experimental validation plan + baselines | §7 | Covered |
| Related work (JEPA family, VLMs, CL, MoE, Mamba, LoRA) | §2 | Covered |
| Licensing / IP | §8 (item 9) | Covered |
| Publication strategy / rebuttals | §9 | Covered |
| Ethics / societal impact | §8 (item 10) | Covered |
| Forgetting bound (theoretical proof sketch) | §4.6 | Covered |
| 2024–2026 citations | Throughout | Covered (V-JEPA 2 2025, LLM-JEPA 2025, LeJEPA 2025, VL-JEPA 2025, surveys 2024–2025) |