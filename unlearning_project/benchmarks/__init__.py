from __future__ import annotations

from pathlib import Path

from .base import BenchmarkAdapter
from .tofu import TOFUAdapter
from .wmdp import WMDPAdapter

ADAPTERS = {
    TOFUAdapter.name: TOFUAdapter,
    WMDPAdapter.name: WMDPAdapter,
}


def create_adapter(config: dict, project_dir: Path) -> BenchmarkAdapter:
    adapter_name = config.get("adapter") or config.get("name")
    if not adapter_name:
        raise ValueError("Benchmark config must define 'adapter' or 'name'")
    if adapter_name not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown benchmark adapter '{adapter_name}'. Known adapters: {known}")
    return ADAPTERS[adapter_name](config=config, project_dir=project_dir)


__all__ = [
    "ADAPTERS",
    "BenchmarkAdapter",
    "TOFUAdapter",
    "WMDPAdapter",
    "create_adapter",
]
