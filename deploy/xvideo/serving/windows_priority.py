"""Set delivery priority inside the real process, after any venv launcher."""
import os


def configure_delivery_priority():
    if os.name != "nt":
        return {"supported": False, "ok": True}
    try:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.SetPriorityClass.restype = wintypes.BOOL
        kernel.GetPriorityClass.argtypes = (wintypes.HANDLE,)
        kernel.GetPriorityClass.restype = wintypes.DWORD
        process = kernel.GetCurrentProcess()
        above_normal = 0x8000
        if not kernel.SetPriorityClass(process, above_normal):
            raise ctypes.WinError(ctypes.get_last_error())
        actual = kernel.GetPriorityClass(process)
        if actual != above_normal:
            raise RuntimeError(f"Requested AboveNormal; received priority {actual}")
        return {"supported": True, "ok": True, "class": "AboveNormal", "value": actual}
    except Exception as exc:
        # Startup diagnostics must not stop the show if Windows denies the change.
        return {"supported": True, "ok": False, "error": str(exc)}
