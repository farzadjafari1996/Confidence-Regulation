"""
Central hyperparameters for confidence-regulation reproduction.

Selection counts (entropy neurons, random controls, null-space k) support both a
percentage and a fixed number; the effective count is min(percentage-derived, fixed).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_DATASET = "stas/c4-en-10k"

GPT2_DEFAULT_MODEL = "gpt2-small"
LLAMA_DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
QWEN_DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"

GPT2_PAPER_ENTROPY_NEURONS = [
    "11.584",
    "11.2378",
    "11.2870",
    "11.2123",
    "11.1611",
    "11.2910",
]


def count_from_pct_and_n(total: int, pct: float, fixed_n: int | None) -> int:
    """Return min(int(pct * total), fixed_n) when fixed_n set; at least 1, capped at total."""
    from_pct = max(1, int(pct * total))
    if fixed_n is None:
        return min(from_pct, total)
    return max(1, min(from_pct, fixed_n, total))


def nullspace_k_from_pct_and_fixed(d_model: int, pct: float, fixed_k: int | None) -> int:
    """Null-space dimension: min(round(pct * d_model), fixed_k) when fixed_k set; at least 1."""
    from_pct = max(1, round(pct * d_model))
    if fixed_k is None:
        return from_pct
    return max(1, min(from_pct, fixed_k))


@dataclass
class Hyperparameters:
    """Tunable defaults shared across phases and model families."""

    # Phase 1 — entropy-neuron selection (last-layer rho ranking)
    entropy_top_pct: float = 0.01
    entropy_top_n: int | None = None

    # Null-space dimension for rho (W_U SVD tail); min with pct * d_model when set
    nullspace_top_pct: float = 0.01
    nullspace_k: int | None = None

    # Data
    dataset: str = DEFAULT_DATASET
    data_end: int = 1000
    max_seq_length: int = 256
    seed: int = 42

    # Phase 2 — ablation
    k: int = 10
    batch_size: int = 1
    n_random: int = 30
    n_random_pct: float | None = None
    chunk_size: int = 20

    # Post-filter on ablation token positions
    filter_start_pos: int = 3
    filter_end_pos: int = -1

    # Plotting
    n_trace_neurons: int = 6
    box_plot_n_random: int = 10
    write_align_n_random: int = 15  # entropy-vs-random unembed alignment plot (10–20 typical)
    write_align_mode: str = "both"  # full | means_only | both

    # Metrics
    use_log2_entropy: bool = False


# Paper-aligned presets (RTX 4070-friendly phase-2 defaults in data_end / k)
GPT2_HPARAMS = Hyperparameters(
    entropy_top_n=12,
    nullspace_top_pct=0.016,
    nullspace_k=12,
    k=6,
    data_end=400,
    n_random=90,
)

LLAMA_HPARAMS = Hyperparameters(
    nullspace_k=20,
    k=4,
    data_end=400,
)

QWEN_08B_HPARAMS = Hyperparameters(
    nullspace_k=10,
    k=4,
    data_end=400,
)

QWEN_2B_HPARAMS = Hyperparameters(
    nullspace_k=20,
    k=4,
    data_end=400,
)

QWEN_HPARAMS = QWEN_08B_HPARAMS

HP_FIELD_NAMES = {f.name for f in fields(Hyperparameters)}
RUN_SETTING_KEYS = {"model", "dtype", "out_dir"}
Family = Literal["gpt2", "llama", "qwen"]

_FAMILY_BASE: dict[Family, Hyperparameters] = {
    "gpt2": GPT2_HPARAMS,
    "llama": LLAMA_HPARAMS,
    "qwen": QWEN_HPARAMS,
}

_FAMILY_DEFAULT_MODEL: dict[Family, str] = {
    "gpt2": GPT2_DEFAULT_MODEL,
    "llama": LLAMA_DEFAULT_MODEL,
    "qwen": QWEN_DEFAULT_MODEL,
}


def qwen_hparams_for_model(model_name: str) -> Hyperparameters:
    """Pick nullspace_k preset from checkpoint size (0.8B vs 2B)."""
    if "2B" in model_name or "2b" in model_name:
        return QWEN_2B_HPARAMS
    return QWEN_08B_HPARAMS


def default_config_path(family: Family) -> pathlib.Path:
    if family == "qwen":
        return CONFIG_DIR / "qwen3_0.8b.yaml"
    return CONFIG_DIR / f"{family}.yaml"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def model_output_dir(model_name: str, base: pathlib.Path | None = None) -> pathlib.Path:
    """Return outputs/<model-slug> (e.g. outputs/gpt2-small)."""
    return (base or DEFAULT_OUTPUT_DIR) / model_slug(model_name)


def hyperparameters_to_dict(hparams: Hyperparameters) -> dict[str, Any]:
    return {name: getattr(hparams, name) for name in HP_FIELD_NAMES}


def hyperparameters_from_dict(data: dict[str, Any], base: Hyperparameters) -> Hyperparameters:
    """Build Hyperparameters from a mapping; unknown keys are ignored."""
    updates: dict[str, Any] = {}
    for key in HP_FIELD_NAMES:
        if key in data:
            updates[key] = data[key]
    return replace(base, **updates)


def load_config_yaml(path: pathlib.Path | str) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    return raw


def _resolve_out_dir(value: Any) -> pathlib.Path:
    if value is None:
        return DEFAULT_OUTPUT_DIR
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_run_settings(
    config_path: pathlib.Path | str | None,
    family: Family,
) -> dict[str, Any]:
    """
    Load YAML run settings for a model family.

    Falls back to config/<family>.yaml when config_path is None, then to the
    in-code Hyperparameters preset if no file exists.
    """
    base = _FAMILY_BASE[family]
    path = pathlib.Path(config_path) if config_path is not None else default_config_path(family)

    if not path.exists():
        return {
            "hparams": base,
            "model": _FAMILY_DEFAULT_MODEL[family],
            "dtype": None,
            "out_dir": DEFAULT_OUTPUT_DIR,
            "config_path": None,
        }

    raw = load_config_yaml(path)
    nested = raw.get("hyperparameters")
    hp_source = nested if isinstance(nested, dict) else raw
    hparams = hyperparameters_from_dict(hp_source, base)

    model = raw.get("model") or _FAMILY_DEFAULT_MODEL[family]
    dtype = raw.get("dtype")
    if isinstance(dtype, str) and dtype.lower() in {"null", "none", ""}:
        dtype = None

    return {
        "hparams": hparams,
        "model": model,
        "dtype": dtype,
        "out_dir": _resolve_out_dir(raw.get("out_dir")),
        "config_path": path.resolve(),
    }


@dataclass
class ModelRunConfig:
    name: str
    entropy_top_pct: float = GPT2_HPARAMS.entropy_top_pct
    entropy_top_n: int | None = GPT2_HPARAMS.entropy_top_n
    nullspace_top_pct: float = GPT2_HPARAMS.nullspace_top_pct
    udark_k: int | None = GPT2_HPARAMS.nullspace_k
    dataset: str = GPT2_HPARAMS.dataset
    k: int = GPT2_HPARAMS.k
    data_end: int = GPT2_HPARAMS.data_end
    batch_size: int = GPT2_HPARAMS.batch_size
    n_random: int = GPT2_HPARAMS.n_random
    n_random_pct: float | None = GPT2_HPARAMS.n_random_pct
    chunk_size: int = GPT2_HPARAMS.chunk_size
    seed: int = GPT2_HPARAMS.seed
    max_seq_length: int = GPT2_HPARAMS.max_seq_length
    filter_start_pos: int = GPT2_HPARAMS.filter_start_pos
    filter_end_pos: int = GPT2_HPARAMS.filter_end_pos
    n_trace_neurons: int = GPT2_HPARAMS.n_trace_neurons
    box_plot_n_random: int = GPT2_HPARAMS.box_plot_n_random
    write_align_n_random: int = GPT2_HPARAMS.write_align_n_random
    write_align_mode: str = GPT2_HPARAMS.write_align_mode
    use_log2_entropy: bool = GPT2_HPARAMS.use_log2_entropy
    dtype: str | None = None
    skip: bool = False

    @property
    def slug(self) -> str:
        return model_slug(self.name)

    def output_dir(self, base: pathlib.Path) -> pathlib.Path:
        return base / self.slug

    def entropy_neuron_count(self, n_neurons: int) -> int:
        return count_from_pct_and_n(n_neurons, self.entropy_top_pct, self.entropy_top_n)

    def random_control_count(self, pool_size: int) -> int:
        fixed_n = self.n_random
        if self.n_random_pct is not None:
            from_pct = max(1, int(self.n_random_pct * pool_size))
            fixed_n = min(fixed_n, from_pct)
        return min(fixed_n, pool_size)

    def nullspace_k(self, d_model: int) -> int:
        return nullspace_k_from_pct_and_fixed(d_model, self.nullspace_top_pct, self.udark_k)

    def udark_start(self, d_model: int) -> int:
        return -self.nullspace_k(d_model)

    def last_layer(self, model) -> int:
        return model.cfg.n_layers - 1

    @classmethod
    def from_hparams(
        cls,
        name: str,
        hparams: Hyperparameters,
        *,
        dtype: str | None = None,
        skip: bool = False,
        **overrides,
    ) -> ModelRunConfig:
        hp_fields = {f.name for f in fields(Hyperparameters)}
        values = {key: getattr(hparams, key) for key in hp_fields}
        values["udark_k"] = values.pop("nullspace_k")
        values.update(overrides)
        cfg_fields = {f.name for f in fields(cls)}
        kwargs = {key: values[key] for key in values if key in cfg_fields}
        kwargs["name"] = name
        kwargs["dtype"] = dtype
        kwargs["skip"] = skip
        return cls(**kwargs)
