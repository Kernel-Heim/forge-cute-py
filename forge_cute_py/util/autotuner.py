from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .bench import do_bench, summarize_times


PACKAGE_NAME = "forge_cute_py"


def _get_home_dir() -> Path:
    return Path(os.getenv(f"{PACKAGE_NAME.upper()}_HOME", Path.home()))


def _default_cache_dir() -> Path:
    env_override = (
        os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_DIR")
        or os.getenv("FORGE_CUTE_PY_CACHE_DIR")
        or os.getenv("KERNELHEIM_CACHE_DIR")
    )
    if env_override:
        return Path(env_override)
    return _get_home_dir() / f".{PACKAGE_NAME}" / "cache"


def _base32(key: str) -> str:
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


@dataclass(frozen=True)
class AutotuneConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def all_kwargs(self) -> Dict[str, Any]:
        return dict(self.kwargs)

    def __str__(self) -> str:
        return f"{self.name}:{self.kwargs}"


class Autotuner:
    def __init__(
        self,
        fn: Callable[..., Any],
        key: Sequence[str],
        configs: Sequence[AutotuneConfig],
        cache_results: bool = True,
        do_bench_fn: Optional[Callable[..., List[float]]] = None,
    ) -> None:
        self.fn = fn
        self.key = list(key)
        self.configs = list(configs) if configs else [AutotuneConfig("default", {})]
        self.cache_results = cache_results
        self._do_bench = do_bench_fn or do_bench
        self._cache: Dict[Tuple[str, ...], AutotuneConfig] = {}

    def _tuning_key(self, kwargs: Dict[str, Any]) -> Tuple[str, ...]:
        key_parts: List[str] = []
        for k in self.key:
            if k in kwargs:
                key_parts.append(str(kwargs[k]))
        for _, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                key_parts.append(str(tuple(arg.shape)))
        return tuple(key_parts)

    def _cache_path(self, tuning_key: Tuple[str, ...]) -> Optional[Path]:
        if not tuning_key:
            return None
        cache_dir = _default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        hash_key = hashlib.sha256("::".join(tuning_key).encode("utf-8")).hexdigest()
        filename = f"{self.fn.__name__}.{_base32(hash_key)}.json"
        return cache_dir / filename

    def _load_cache(self, path: Path) -> Optional[AutotuneConfig]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config_name = payload.get("best_config")
        config_kwargs = payload.get("best_kwargs", {})
        if config_name is None:
            return None
        return AutotuneConfig(config_name, config_kwargs)

    def _store_cache(self, path: Path, best: AutotuneConfig, timings: Dict[str, float]) -> None:
        payload = {
            "best_config": best.name,
            "best_kwargs": best.kwargs,
            "timings_ms": timings,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def __call__(self, *args, **kwargs):
        tuning_key = self._tuning_key(kwargs)
        cache_path = self._cache_path(tuning_key) if self.cache_results else None
        if cache_path is not None:
            cached = self._load_cache(cache_path)
            if cached is not None:
                return self.fn(*args, **cached.all_kwargs(), **kwargs)

        timings: Dict[str, float] = {}
        best_config = self.configs[0]
        best_time = float("inf")
        for config in self.configs:

            def call():
                return self.fn(*args, **config.all_kwargs(), **kwargs)

            times = self._do_bench(call, warmup=5, rep=25)
            stats = summarize_times(times)
            timings[str(config)] = stats["p50_ms"]
            if stats["p50_ms"] < best_time:
                best_time = stats["p50_ms"]
                best_config = config

        if cache_path is not None:
            self._store_cache(cache_path, best_config, timings)

        return self.fn(*args, **best_config.all_kwargs(), **kwargs)
