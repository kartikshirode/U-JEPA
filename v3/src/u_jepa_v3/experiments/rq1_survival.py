"""RQ1: does poison survive an automated maintenance pipeline and its corrections.

One arm is one (model, editor, base_rate, revert_lag, seed) cell. The arm streams
a simulated feed through the editor and probes at intervals. Nothing decides
anything here; the gate arrives in stage 2. This measures the undefended
pipeline, which is the baseline the gate has to beat.

The measurement that matters is not "did the edit take". It is what the model
believes about a poisoned fact AFTER the upstream correction has been applied in
good faith, under three levels of questioning pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..data.feed import poison_state
from ..editors.base import Editor, Responder
from ..probes.downstream import downstream_harm
from ..probes.efficacy import efficacy, locality
from ..probes.elicitation import elicitation_rate
from ..probes.general_ability import general_ability
from ..runs.state import RunState
from ..schema import FeedEntry

PROBE_DIR_ENV = "U_JEPA_V3_PROBE_DIR"


@dataclass(frozen=True)
class Rq1Config:
    checkpoint_every: int
    seed: int
    model: str
    editor: str
    base_rate: float
    revert_lag: int

    def __post_init__(self) -> None:
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {self.checkpoint_every}")


def _checkpoint(
    editor: Editor,
    feed: list[FeedEntry],
    upto: int,
    suites: dict,
    locality_pairs: list[tuple[str, str]],
    n_failed: int,
    config: Rq1Config,
    hop_questions: dict[str, list[tuple[str, str, str]]],
) -> dict:
    responder = editor.responder()
    uncorrected, corrected = poison_state(feed, upto)
    benign = [e.candidate for e in feed[:upto] if not e.is_poison and not e.reverts]

    # Dependent questions for poison the pipeline has actually reached, whether or
    # not its correction has landed. Empty until hop_questions is supplied.
    reached = {e.entry_id for e in uncorrected + corrected}
    hops = [q for eid, qs in hop_questions.items() if eid in reached for q in qs]
    harm = downstream_harm(responder, hops)

    return {
        "at": upto,
        "seed": config.seed,
        "n_failed": n_failed,
        "n_poison_uncorrected": len(uncorrected),
        "n_poison_corrected": len(corrected),
        "benign_efficacy": efficacy(responder, benign),
        "poison_uncorrected": efficacy(responder, [e.candidate for e in uncorrected]),
        "poison_corrected_direct": elicitation_rate(
            responder, [e.candidate for e in corrected], mode="direct"),
        "poison_corrected_paraphrase": elicitation_rate(
            responder, [e.candidate for e in corrected], mode="paraphrase"),
        "poison_corrected_leading": elicitation_rate(
            responder, [e.candidate for e in corrected], mode="leading"),
        "downstream_corrupted": harm.corrupted,
        "downstream_poisoned": harm.poisoned_answer,
        "n_hop_questions": harm.n_questions,
        "locality": locality(responder, locality_pairs),
        "general_mean": general_ability(responder, suites).mean,
    }


def run_arm(
    editor: Editor,
    feed: list[FeedEntry],
    suites: dict[str, list[tuple[str, str]]],
    locality_pairs: list[tuple[str, str]],
    base_responder: Responder,
    config: Rq1Config,
    hop_questions: dict[str, list[tuple[str, str, str]]] | None = None,
) -> RunState:
    """Stream one feed through one editor, probing every checkpoint_every entries.

    hop_questions maps a poison entry_id to (prompt, true_answer, poison_answer)
    triples that depend on that fact. Empty means the downstream numbers come
    back zero rather than the probe being skipped, so the field is always there
    for the analysis.
    """
    hop_questions = hop_questions or {}
    state = RunState(cell_id="", meta={
        "model": config.model,
        "editor": config.editor,
        "seed": config.seed,
        "base_rate": config.base_rate,
        "revert_lag": config.revert_lag,
        "n_feed": len(feed),
        # The untouched-base arm, measured once before anything is applied.
        "baseline_general": general_ability(base_responder, suites).mean,
    })

    n_failed = 0
    for start in range(0, len(feed), config.checkpoint_every):
        batch = feed[start : start + config.checkpoint_every]
        results = editor.apply([e.candidate for e in batch])
        n_failed += sum(not r.succeeded for r in results)
        upto = start + len(batch)
        state.checkpoints.append(
            _checkpoint(editor, feed, upto, suites, locality_pairs, n_failed,
                        config, hop_questions)
        )

    state.finished = True
    return state


def _probe_dir():
    """Directory holding the pinned probe sets.

    Pinned rather than downloaded at run time, because a probe set that shifts
    between cells makes every cross-cell comparison meaningless and does it
    silently.
    """
    import os
    from pathlib import Path

    raw = os.environ.get(PROBE_DIR_ENV)
    if not raw:
        raise RuntimeError(
            f"{PROBE_DIR_ENV} is unset. Point it at a directory holding "
            "sst.json, mmlu.json, mrpc.json, nli.json and locality.json, each a "
            "JSON list of [prompt, expected_answer] pairs."
        )
    return Path(raw)


def _read_pairs(name: str) -> list[tuple[str, str]]:
    import json

    path = _probe_dir() / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"probe set {name} not found at {path}")
    return [(str(p), str(a)) for p, a in json.loads(path.read_text(encoding="utf-8"))]


def _load_suites() -> dict[str, list[tuple[str, str]]]:
    from ..probes.general_ability import REQUIRED_SUITES

    return {name: _read_pairs(name) for name in REQUIRED_SUITES}


def _load_locality() -> list[tuple[str, str]]:
    return _read_pairs("locality")


def _load_base(model_name: str):
    """The untouched model and tokenizer, for the control arm.

    Left padding because these are generation calls; right padding puts the pad
    tokens where the answer should start and quietly returns empty strings.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..env import device_capability, preferred_dtype_str

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[preferred_dtype_str(device_capability())]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": 0})
    model.eval()
    return model, tok


def run_cell_from_params(params: dict) -> RunState:
    """Build and run one RQ1 cell from grid params. Called by the shard worker.

    Expected keys: model, editor, hparams, seed, base_rate, revert_lag,
    n_benign, n_poison, attack_family, checkpoint_every.
    """
    from ..data import adversarial, wikibigedit
    from ..data.feed import build_feed
    from ..editors import registry
    from ..editors.easyedit_adapter import HFResponder

    registry.register_defaults()

    benign = wikibigedit.load_candidates(n=params["n_benign"], seed=params["seed"])
    family = params["attack_family"]
    if family == adversarial.AttackFamily.OBJECT_SWAP.value:
        pairs = adversarial.poison_object_swap(benign, params["seed"], params["n_poison"])
    elif family == adversarial.AttackFamily.TYPE_CONSISTENT.value:
        pairs = adversarial.poison_type_consistent(benign, params["seed"], params["n_poison"])
    else:
        history = adversarial.build_history(benign)
        pairs = adversarial.poison_temporal_stale(history, params["seed"], params["n_poison"])

    feed = build_feed(benign, pairs, params["base_rate"], params["revert_lag"], params["seed"])
    editor = registry.build(params["editor"], hparams_path=params["hparams"])

    base_model, base_tok = _load_base(params["model"])
    base_responder = HFResponder(base_model, base_tok)

    config = Rq1Config(
        checkpoint_every=params["checkpoint_every"], seed=params["seed"],
        model=params["model"], editor=params["editor"],
        base_rate=params["base_rate"], revert_lag=params["revert_lag"],
    )
    return run_arm(editor, feed, _load_suites(), _load_locality(), base_responder, config)
