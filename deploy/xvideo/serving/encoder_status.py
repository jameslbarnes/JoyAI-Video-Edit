"""Read-only encoder placement evidence; never moves or synchronizes weights."""
import os


def encoder_status(pipeline):
    if pipeline is None:
        return {"loaded": False}
    placements = {}
    dtypes = set()
    for parameter in pipeline.text_encoder.parameters():
        device = str(parameter.device)
        placements[device] = placements.get(device, 0) + parameter.numel()
        dtypes.add(str(parameter.dtype))
    execution = str(pipeline.text_execution_device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    index = int(execution.split(":")[-1]) if execution.startswith("cuda:") else None
    return {"loaded": True, "executionDevice": execution,
            "physicalGpu": visible[index] if index is not None and index < len(visible) else None,
            "parameterElementsByDevice": placements, "dtypes": sorted(dtypes),
            "fullyResident": bool(placements) and set(placements) == {execution}}
