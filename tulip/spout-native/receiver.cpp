// Snapshot the newest Spout texture into caller-owned CUDA memory. No CPU pixels.
// The receiver is used serially, from Python's input thread pool. Each call
// installs the originating CUDA context; no assumption about thread affinity.
#include <windows.h>
#include <d3d11.h>
#include <d3d10.h>
#include <dxgi.h>
#include <cuda.h>
#include <cudaD3D11.h>
#include <SpoutDX.h>
#include <cstdint>
#include <stdexcept>
#include <string>

static void ck(CUresult r) {
  if (r == CUDA_SUCCESS) return;
  const char* s=nullptr; cuGetErrorString(r,&s);
  throw std::runtime_error(s ? s : "CUDA input interop failure");
}
static void hr(HRESULT r) {
  if (FAILED(r)) throw std::runtime_error("D3D11 input failure: "+std::to_string(r));
}
struct Context {
  explicit Context(CUcontext ctx) { ck(cuCtxPushCurrent(ctx)); }
  ~Context() { CUcontext old; cuCtxPopCurrent(&old); }
};
struct Input {
  CUcontext cuda=nullptr;
  CUstream stream=nullptr;
  ID3D11Device* device=nullptr;
  ID3D11DeviceContext* context=nullptr;
  ID3D11Texture2D* texture=nullptr;
  CUgraphicsResource resource=nullptr;
  spoutDX spout;
  std::string name, error;
  unsigned width=0,height=0,format=0;
  int64_t copied_frame=-1;
  bool pending=false;
  Input(const char* source) : name(source ? source : "") {
    if (name.empty()) throw std::runtime_error("An explicit Spout input name is required");
    try {
      ck(cuCtxGetCurrent(&cuda));
      if (!cuda) throw std::runtime_error("A current CUDA input context is required");
      CUdevice selected; ck(cuCtxGetDevice(&selected));
      IDXGIFactory1* factory=nullptr;
      hr(CreateDXGIFactory1(__uuidof(IDXGIFactory1),(void**)&factory));
      for (UINT i=0; ; ++i) {
        IDXGIAdapter1* adapter=nullptr;
        if (factory->EnumAdapters1(i,&adapter)==DXGI_ERROR_NOT_FOUND) break;
        CUdevice candidate=-1;
        if (cuD3D11GetDevice(&candidate,adapter)==CUDA_SUCCESS && candidate==selected) {
          D3D_FEATURE_LEVEL level;
          HRESULT result=D3D11CreateDevice(adapter,D3D_DRIVER_TYPE_UNKNOWN,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context);
          adapter->Release(); factory->Release(); factory=nullptr;
          hr(result); break;
        }
        adapter->Release();
      }
      if (factory) factory->Release();
      if (!device) throw std::runtime_error("No D3D11 adapter matches the CUDA input GPU");
      ID3D10Multithread* protect=nullptr;
      hr(context->QueryInterface(__uuidof(ID3D10Multithread),(void**)&protect));
      protect->SetMultithreadProtected(TRUE); protect->Release();
      if (!spout.OpenDirectX11(device)) throw std::runtime_error("Spout input initialization failed");
      // Never silently create a device on a different GPU.
      spout.SetAdapterAuto(false);
      spout.frame.SetFrameCount(true);
      spout.SetReceiverName(name.c_str());
      ck(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
    } catch (...) { release(); throw; }
  }
  void clear_texture() {
    if (resource) { cuGraphicsUnregisterResource(resource); resource=nullptr; }
    if (texture) { texture->Release(); texture=nullptr; }
  }
  void release() {
    clear_texture();
    spout.ReleaseReceiver();
    spout.CloseDirectX11();
    if (stream) { cuStreamDestroy(stream); stream=nullptr; }
    if (context) { context->Release(); context=nullptr; }
    if (device) { device->Release(); device=nullptr; }
  }
  ~Input() { if (cuda) { Context guard(cuda); release(); } }
  // 0: not fresh, 1: copied, 2: destination too small, -1: error.
  int read(CUdeviceptr dst, uint64_t capacity, uint64_t* info) {
    Context guard(cuda);
    if (!spout.ReceiveTexture(&texture)) return 0;
    if (spout.IsUpdated()) {
      clear_texture();
      copied_frame=-1;
      pending=false;
      width=spout.GetSenderWidth(); height=spout.GetSenderHeight();
      format=(unsigned)spout.GetSenderFormat();
      if (!width || !height || width>4096 || height>2160 ||
          (format!=DXGI_FORMAT_R8G8B8A8_UNORM && format!=DXGI_FORMAT_B8G8R8A8_UNORM))
        throw std::runtime_error("Unsupported Spout input geometry or format");
      D3D11_TEXTURE2D_DESC desc{};
      desc.Width=width;desc.Height=height;desc.MipLevels=1;desc.ArraySize=1;
      desc.Format=(DXGI_FORMAT)format;desc.SampleDesc.Count=1;
      desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
      hr(device->CreateTexture2D(&desc,nullptr,&texture));
      ck(cuGraphicsD3D11RegisterResource(&resource,texture,CU_GRAPHICS_REGISTER_FLAGS_NONE));
      // The SDK returns before copying on a geometry update.
      return 0;
    }
    info[0]=width;info[1]=height;info[2]=format;info[3]=spout.GetSenderFrame();
    if (spout.IsFrameNew()) pending=true;
    if (!resource || !pending || (int64_t)info[3]==copied_frame) return 0;
    if (capacity<uint64_t(width)*height*4) return 2;
    ck(cuGraphicsMapResources(1,&resource,stream));
    try {
      CUarray array;ck(cuGraphicsSubResourceGetMappedArray(&array,resource,0,0));
      CUDA_MEMCPY2D copy{};
      copy.srcMemoryType=CU_MEMORYTYPE_ARRAY;copy.srcArray=array;
      copy.dstMemoryType=CU_MEMORYTYPE_DEVICE;copy.dstDevice=dst;
      copy.dstPitch=uint64_t(width)*4;copy.WidthInBytes=copy.dstPitch;copy.Height=height;
      ck(cuMemcpy2DAsync(&copy,stream));
    } catch (...) {
      cuGraphicsUnmapResources(1,&resource,stream);cuStreamSynchronize(stream);throw;
    }
    ck(cuGraphicsUnmapResources(1,&resource,stream));
    ck(cuStreamSynchronize(stream)); // Caller can retain/reuse its own storage.
    copied_frame=(int64_t)info[3];
    pending=false;
    return 1;
  }
};
static thread_local std::string last_error;
extern "C" {
__declspec(dllexport) void* jsi_create(const char* name) {
  try { return new Input(name); } catch (const std::exception& e) { last_error=e.what();return nullptr; }
}
__declspec(dllexport) int jsi_read(void* p, uint64_t dst, uint64_t bytes, uint64_t* info) {
  if (!p || !info) return -1;
  try { return static_cast<Input*>(p)->read(dst,bytes,info); }
  catch (const std::exception& e) { static_cast<Input*>(p)->error=e.what();return -1; }
}
__declspec(dllexport) const char* jsi_error(void* p) {
  return p ? static_cast<Input*>(p)->error.c_str() : last_error.c_str();
}
__declspec(dllexport) void jsi_close(void* p) { delete static_cast<Input*>(p); }
}
