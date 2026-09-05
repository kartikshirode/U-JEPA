"""Check a hparams file against the installed EasyEdit and the real model config.

The YAML files in v3/hparams/ were written from published templates on a laptop
with no EasyEdit and no model. That is enough to start from and not enough to
trust, so this turns them from a guess into something that either passes or
names what is wrong.

    python v3/scripts/04_check_hparams.py v3/hparams/

Three checks. The HyperParams class for the algorithm has to build from the
file, which catches an invented or missing field. Every layer index has to be
inside the model. And the module templates have to name modules the model
actually has.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEMPLATE_KEYS = ("rewrite_module_tmp", "layer_module_tmp", "mlp_module_tmp",
                 "attn_module_tmp")


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_builds(path: Path, alg: str) -> list[str]:
    """The real test: EasyEdit's own loader either accepts the file or does not."""
    try:
        from easyeditor import get_hparams
    except ImportError:
        return ["easyeditor is not installed, so the field names are still unchecked"]

    try:
        get_hparams(alg.lower(), str(path))
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"]
    return []


def check_layers(raw: dict, n_layers: int) -> list[str]:
    problems = []
    for key in ("layers", "v_loss_layer"):
        value = raw.get(key)
        if value is None:
            continue
        indices = value if isinstance(value, list) else [value]
        for index in indices:
            if not 0 <= int(index) < n_layers:
                problems.append(
                    f"{key} names layer {index}, outside this model's 0 to {n_layers - 1}")
    return problems


def check_modules(raw: dict, model_name: str) -> list[str]:
    """Format each template against a real layer index and look the module up."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        return ["transformers is not installed, so module names are unchecked"]

    try:
        AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", low_cpu_mem_usage=True)
    except Exception as exc:
        return [f"could not load {model_name} to check module names "
                f"({type(exc).__name__}); run this on the login node with the "
                "weights already cached"]

    names = {name for name, _ in model.named_modules()}
    problems = []
    for key in TEMPLATE_KEYS + ("ln_f_module", "lm_head_module"):
        template = raw.get(key)
        if not isinstance(template, str):
            continue
        resolved = template.format(0) if "{}" in template else template
        if resolved not in names:
            problems.append(f"{key} resolves to {resolved!r}, which this model has no module for")
    return problems


def check_file(path: Path, skip_model: bool) -> bool:
    print(f"\n{path.name}")
    raw = load_yaml(path)
    alg = raw.get("alg_name")
    model_name = raw.get("model_name")
    if not alg or not model_name:
        print("  FAIL alg_name and model_name are both required")
        return False
    print(f"  alg {alg}, model {model_name}")

    problems = check_builds(path, alg)

    n_layers = None
    if not skip_model:
        try:
            from transformers import AutoConfig

            n_layers = AutoConfig.from_pretrained(model_name).num_hidden_layers
        except Exception as exc:
            problems.append(f"could not read the model config ({type(exc).__name__}: {exc})")
    if n_layers:
        print(f"  model has {n_layers} layers")
        problems.extend(check_layers(raw, n_layers))
        problems.extend(check_modules(raw, model_name))

    for problem in problems:
        print(f"  PROBLEM {problem}")
    if not problems:
        print("  ok")
    return not problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="a hparams file or a directory of them")
    parser.add_argument("--skip-model", action="store_true",
                        help="check field names only, without downloading the model")
    args = parser.parse_args(argv)

    target = Path(args.target)
    files = sorted(target.glob("*.yaml")) if target.is_dir() else [target]
    if not files:
        print(f"no yaml files under {target}")
        return 1

    results = [check_file(path, args.skip_model) for path in files]
    print("\n" + "=" * 60)
    print(f"{sum(results)}/{len(results)} files clean")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
