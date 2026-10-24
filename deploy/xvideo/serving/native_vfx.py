"""Small ctypes binding to the pinned Windows VFX SDK's public C API.

NvCVImage_Init wraps caller-owned CUDA byte buffers; the SDK never owns their
storage. ABI: NvCVImage API / VFX SDK 1.2, Windows x64 (64-byte descriptor).
See https://docs.nvidia.com/maxine/nvcvimage/1.0.0/functions.html and
https://docs.nvidia.com/maxine/vfx/latest/API/Reference/VideoEffectsFunctions.html.
"""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import sys


class NvCVImage(C.Structure):
    _fields_ = [
        ("width", C.c_uint), ("height", C.c_uint), ("pitch", C.c_int),
        ("pixelFormat", C.c_int), ("componentType", C.c_int),
        ("pixelBytes", C.c_ubyte), ("componentBytes", C.c_ubyte),
        ("numComponents", C.c_ubyte), ("planar", C.c_ubyte),
        ("gpuMem", C.c_ubyte), ("colorspace", C.c_ubyte),
        ("reserved", C.c_ubyte * 2), ("pixels", C.c_void_p),
        ("deletePtr", C.c_void_p), ("deleteProc", C.c_void_p),
        ("bufferBytes", C.c_ulonglong),
    ]


class VfxApi:
    def __init__(self):
        if sys.platform != "win32" or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError("Direct VFX currently requires Windows x64")
        if C.sizeof(NvCVImage) != 64 or NvCVImage.pixels.offset != 32:
            raise RuntimeError("Unexpected NvCVImage ABI")
        package = Path(__file__).resolve().parents[3] / "tulip/research/vfx-probe"
        if not (package / "nvvfx").is_dir():
            raise RuntimeError("Pinned NVIDIA VFX installation is missing")
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        import nvvfx
        if nvvfx.__version__ != "0.1.0.1" or nvvfx.get_sdk_version() != "1.2.0":
            raise RuntimeError("Direct VFX requires tested SDK 1.2.0 / wheel 0.1.0.1")
        self.version = nvvfx.get_sdk_version()
        libs = package / "nvvfx/libs"
        # CDLL releases the GIL around the public C calls.
        self.vfx = C.CDLL(str(libs / "NVVideoEffects.dll"))
        self.image = C.CDLL(str(libs / "NVCVImage.dll"))
        ptr, image = C.c_void_p, C.POINTER(NvCVImage)
        self.create = self._bind(self.vfx, "NvVFX_CreateEffect", [C.c_char_p, C.POINTER(ptr)])
        self.destroy = self._bind(self.vfx, "NvVFX_DestroyEffect", [ptr], None)
        self.set_u32 = self._bind(self.vfx, "NvVFX_SetU32", [ptr, C.c_char_p, C.c_uint])
        self.set_image = self._bind(self.vfx, "NvVFX_SetImage", [ptr, C.c_char_p, image])
        self.set_stream = self._bind(self.vfx, "NvVFX_SetCudaStream", [ptr, C.c_char_p, ptr])
        self.load = self._bind(self.vfx, "NvVFX_Load", [ptr])
        self.run = self._bind(self.vfx, "NvVFX_Run", [ptr, C.c_int])
        self.init_image = self._bind(self.image, "NvCVImage_Init",
            [image, C.c_uint, C.c_uint, C.c_int, ptr, C.c_int, C.c_int, C.c_uint, C.c_uint])
        self.error_string = self._bind(self.image, "NvCV_GetErrorStringFromCode",
                                      [C.c_int], C.c_char_p)

    @staticmethod
    def _bind(lib, name, args, result=C.c_int):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
        return fn

    def check(self, code, operation):
        if code:
            message = self.error_string(code)
            message = message.decode("utf-8", "replace") if message else "unknown error"
            raise RuntimeError(f"{operation}: {message} ({code})")

    def wrap_rgba(self, tensor):
        if tensor.ndim != 3 or tensor.shape[2] != 4 or not tensor.is_contiguous():
            raise ValueError("VFX requires contiguous HWC RGBA storage")
        if tensor.element_size() != 1 or tensor.device.type != "cuda":
            raise ValueError("VFX requires CUDA byte storage")
        image = NvCVImage()
        # NVCV_RGBA=6, NVCV_U8=1, NVCV_CHUNKY=0, NVCV_CUDA=1.
        self.check(self.init_image(C.byref(image), tensor.shape[1], tensor.shape[0],
            tensor.stride(0), tensor.data_ptr(), 6, 1, 0, 1), "NvCVImage_Init")
        if (image.pixels != tensor.data_ptr() or image.pixelBytes != 4 or
                image.gpuMem != 1 or image.deletePtr or image.deleteProc):
            raise RuntimeError("SDK did not preserve the caller-owned RGBA descriptor")
        return image


@lru_cache(maxsize=1)
def api():
    return VfxApi()
