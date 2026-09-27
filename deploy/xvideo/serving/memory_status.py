"""Read-only allocator counters; never synchronize, allocate tensors, or trim caches."""
import os
import threading
import sys
import traceback


def thread_stacks():
    """Bounded, read-only Python stacks; no local variables or GPU synchronization."""
    names = {t.ident: t.name for t in threading.enumerate()}
    return [{'id': ident, 'name': names.get(ident, 'unknown'),
             'stack': [{'file': f.filename, 'line': f.lineno, 'function': f.name}
                       for f in traceback.extract_stack(frame, limit=32)]}
            for ident, frame in list(sys._current_frames().items())[:128]]


def memory_status(runtime=None, *, cuda=None):
    if cuda is None:
        import torch
        cuda = torch.cuda
    result = {'initialized': cuda.is_initialized(), 'pythonThreads': threading.active_count(),
              'graphRunners': len(getattr(runtime, 'graph_runners', {})),
              'allocatorConfig': os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')),
              'devices': []}
    if not result['initialized']:
        return result
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    for index in range(cuda.device_count()):
        stats = cuda.memory_stats(index)
        result['devices'].append({
            'device': f'cuda:{index}',
            'physicalGpu': visible[index] if index < len(visible) else None,
            'allocatedMiB': stats.get('allocated_bytes.all.current', 0) / 2**20,
            'reservedMiB': stats.get('reserved_bytes.all.current', 0) / 2**20,
            'activeMiB': stats.get('active_bytes.all.current', 0) / 2**20,
            'inactiveSplitMiB': stats.get('inactive_split_bytes.all.current', 0) / 2**20,
            'peakAllocatedMiB': stats.get('allocated_bytes.all.peak', 0) / 2**20,
            'allocationRetries': stats.get('num_alloc_retries', 0),
            'outOfMemoryCount': stats.get('num_ooms', 0),
            'deviceAllocations': stats.get('num_device_alloc', 0),
            'deviceFrees': stats.get('num_device_free', 0),
        })
    return result
