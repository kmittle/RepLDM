"""Inference pipeline exports loaded only when requested.

Keeping the package initializer lazy prevents an SDXL-only run from executing
the unrelated ControlNet and FreeScale pipelines during package import.
"""

from importlib import import_module


_EXPORTS = {
    "RepLDMSDXLPipeline": (
        "InferencePipelines.RepLDM.pipeline_repldm_sdxl",
        "RepLDMSDXLPipeline",
    ),
    "RepLDMSDXLControlNetPipeline": (
        "InferencePipelines.RepLDM.pipeline_repldm_sdxl_controlnet",
        "RepLDMSDXLControlNetPipeline",
    ),
    "FreeScaleSDXLPipeline": (
        "InferencePipelines.FreeScale.pipeline_freescale_sdxl",
        "FreeScaleSDXLPipeline",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
