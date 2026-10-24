// GPU-owned staging ring, CUDA/D3D11 interop, and an independent presentation
// thread. Python never owns the presentation deadline or a queued texture.
#include <windows.h>
#include <avrt.h>
#include <d3d11.h>
#include <d3d10.h>
#include <dxgi.h>
#include <cuda.h>
#include <cudaD3D11.h>
#include <SpoutDX.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "PresentTrace.h"
#include "FrameTrace.h"

using Clock = std::chrono::steady_clock;
static double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
static double wall() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
static void ck(CUresult r) {
  if (r == CUDA_SUCCESS) return;
  const char* s = nullptr; cuGetErrorString(r, &s);
  throw std::runtime_error(s ? s : "CUDA interop failure");
}
static void hr(HRESULT r, const char* what) {
  if (FAILED(r)) throw std::runtime_error(std::string(what) + ": " + std::to_string(r));
}
struct Stats {
  uint64_t submitted=0, sent=0, queued=0, discarded=0, underruns=0;
  uint64_t missed_ticks=0, access_misses=0, last_token=0, high_resolution=0;
  double last_send_at=0, send_ms=0, copy_ms=0, late_ms=0, max_gap_ms=0;
  double first_send_at=0, gap_sum_ms=0, gap_square_sum=0;
};
// Separate, size-checked diagnostics ABI: never enlarge the original Stats
// written into existing callers' buffers. These are CPU call spans, not CUDA
// event timings, and add no synchronization to the copy path.
struct CopyTrace {
  uint64_t token=0;
  double at=0, context_ms=0, event_wait_ms=0, map_ms=0, array_ms=0;
  double enqueue_ms=0, unmap_ms=0, synchronize_ms=0, total_ms=0;
  double source_ready=0, gpu_span_ms=0, priority=0;
};
struct Slot {
  ID3D11Texture2D* texture=nullptr;
  CUgraphicsResource resource=nullptr;
  uint64_t token=0;
};
struct Context {
  CUcontext previous=nullptr;
  explicit Context(CUcontext ctx) { ck(cuCtxPushCurrent(ctx)); }
  ~Context() { cuCtxPopCurrent(&previous); }
};
struct Sender {
  CUcontext cuda=nullptr;
  CUstream copy_stream=nullptr;
  CUevent copy_begin=nullptr, copy_end=nullptr;
  int copy_priority=0;
  ID3D11Device* device=nullptr;
  ID3D11DeviceContext* d3d=nullptr;
  spoutDX spout;
  std::vector<Slot> slots;
  std::deque<size_t> free, ready;
  std::mutex mutex;
  std::condition_variable changed;
  std::thread worker;
  HANDLE stop_event=nullptr, timer=nullptr;
  bool stopping=false;
  std::string error;
  Stats stats;
  CopyTrace copy_trace;
  PresentTrace present_trace;
  FrameTrace frame_trace;
  unsigned width, height, prime;
  double fps;

  Sender(const char* name, unsigned w, unsigned h, double rate, unsigned capacity, unsigned prefill)
    : width(w), height(h), prime(prefill), fps(rate) {
    if (!name || !*name || w<2 || h<2 || w>4096 || h>2160 ||
        rate<1 || rate>64 || capacity<2 || capacity>32 || !prefill || prefill>capacity)
      throw std::runtime_error("Invalid Spout geometry, clock or queue");
    try {
      ck(cuCtxGetCurrent(&cuda));
      if (!cuda) throw std::runtime_error("A current CUDA context is required");
      CUdevice selected; ck(cuCtxGetDevice(&selected));
      IDXGIFactory1* factory=nullptr; hr(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory), "DXGI factory");
      for (UINT i=0; ; ++i) {
        IDXGIAdapter1* adapter=nullptr;
        if (factory->EnumAdapters1(i, &adapter)==DXGI_ERROR_NOT_FOUND) break;
        CUdevice candidate=-1;
        if (cuD3D11GetDevice(&candidate, adapter)==CUDA_SUCCESS && candidate==selected) {
          D3D_FEATURE_LEVEL level;
          HRESULT result=D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION, &device, &level, &d3d);
          adapter->Release(); factory->Release(); factory=nullptr;
          hr(result, "Matching D3D11 device"); break;
        }
        adapter->Release();
      }
      if (factory) factory->Release();
      if (!device) throw std::runtime_error("No D3D11 adapter matches the CUDA output GPU");
      ID3D10Multithread* protection=nullptr;
      hr(d3d->QueryInterface(__uuidof(ID3D10Multithread), (void**)&protection), "D3D thread protection");
      protection->SetMultithreadProtected(TRUE); protection->Release();
      if (!spout.OpenDirectX11(device)) throw std::runtime_error("Spout DirectX initialization failed");
      spout.frame.SetFrameCount(true);
      spout.SetSenderName(name);
      spout.SetSenderFormat(DXGI_FORMAT_R8G8B8A8_UNORM);
      ck(cuStreamCreate(&copy_stream, CU_STREAM_NON_BLOCKING));
      ck(cuEventCreate(&copy_begin, CU_EVENT_DEFAULT));
      ck(cuEventCreate(&copy_end, CU_EVENT_DEFAULT));
      slots.resize(capacity);
      D3D11_TEXTURE2D_DESC desc{};
      desc.Width=w; desc.Height=h; desc.MipLevels=1; desc.ArraySize=1;
      desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM; desc.SampleDesc.Count=1;
      desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
      for (size_t i=0; i<slots.size(); ++i) {
        hr(device->CreateTexture2D(&desc, nullptr, &slots[i].texture), "Staging texture");
        ck(cuGraphicsD3D11RegisterResource(&slots[i].resource, slots[i].texture,
          CU_GRAPHICS_REGISTER_FLAGS_NONE));
        free.push_back(i);
      }
      stop_event=CreateEventW(nullptr, TRUE, FALSE, nullptr);
      timer=CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
      stats.high_resolution=timer ? 1 : 0;
      if (!timer) timer=CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
      if (!stop_event || !timer) throw std::runtime_error("Presentation timer creation failed");
      worker=std::thread(&Sender::run, this);
    } catch (...) { release(); throw; }
  }
  ~Sender() { release(); }
  void release() {
    { std::lock_guard<std::mutex> lock(mutex); stopping=true; changed.notify_all(); }
    if (stop_event) SetEvent(stop_event);
    if (worker.joinable()) worker.join();
    if (cuda) {
      CUcontext prior=nullptr; cuCtxPushCurrent(cuda);
      if (copy_stream) cuStreamSynchronize(copy_stream);
      for (auto& s:slots) {
        if (s.resource) cuGraphicsUnregisterResource(s.resource);
        if (s.texture) s.texture->Release();
        s={};
      }
      if (copy_stream) cuStreamDestroy(copy_stream);
      if (copy_begin) cuEventDestroy(copy_begin);
      if (copy_end) cuEventDestroy(copy_end);
      copy_stream=nullptr; cuCtxPopCurrent(&prior);
    }
    spout.ReleaseSender(); spout.CloseDirectX11();
    if (d3d) { d3d->Release(); d3d=nullptr; }
    if (device) { device->Release(); device=nullptr; }
    if (timer) { CloseHandle(timer); timer=nullptr; }
    if (stop_event) { CloseHandle(stop_event); stop_event=nullptr; }
  }
  // The feeder alone calls this between completed submissions. No presentation
  // clock, texture order or reserve changes; priority is merely a CUDA hint.
  void set_copy_priority(int requested) {
    if (requested!=0 && requested!=-1) throw std::runtime_error("Copy priority must be 0 or -1");
    if (requested==copy_priority) return;
    Context context(cuda);
    CUstream replacement=nullptr;
    ck(cuStreamCreateWithPriority(&replacement, CU_STREAM_NON_BLOCKING, requested));
    ck(cuStreamDestroy(copy_stream));
    copy_stream=replacement;
    ck(cuStreamGetPriority(copy_stream, &copy_priority));
  }
  int submit(CUdeviceptr pixels, CUevent event, uint64_t token, bool replace=false) {
    size_t index;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (stopping || !error.empty()) return -1;
      if (free.empty()) return 0;
      index=free.front(); free.pop_front();
    }
    auto start=Clock::now();
    CopyTrace trace;
    trace.token=token; trace.at=wall();
    trace.priority=copy_priority;
    auto phase=start;
    auto elapsed=[&] {
      auto now=Clock::now(); double ms=seconds(now-phase)*1000;
      phase=now; return ms;
    };
    auto& slot=slots[index];
    bool mapped=false;
    try {
      Context context(cuda);
      trace.context_ms=elapsed();
      CUresult source_status=event ? cuEventQuery(event) : CUDA_SUCCESS;
      if (source_status!=CUDA_ERROR_NOT_READY) ck(source_status);
      trace.source_ready=source_status==CUDA_SUCCESS ? 1 : 0;
      if (event) ck(cuStreamWaitEvent(copy_stream, event, 0));
      ck(cuEventRecord(copy_begin, copy_stream));
      trace.event_wait_ms=elapsed();
      ck(cuGraphicsMapResources(1, &slot.resource, copy_stream)); mapped=true;
      trace.map_ms=elapsed();
      CUarray array; ck(cuGraphicsSubResourceGetMappedArray(&array, slot.resource, 0, 0));
      trace.array_ms=elapsed();
      CUDA_MEMCPY2D copy{};
      copy.srcMemoryType=CU_MEMORYTYPE_DEVICE; copy.srcDevice=pixels; copy.srcPitch=width*4;
      copy.dstMemoryType=CU_MEMORYTYPE_ARRAY; copy.dstArray=array;
      copy.WidthInBytes=width*4; copy.Height=height;
      ck(cuMemcpy2DAsync(&copy, copy_stream));
      trace.enqueue_ms=elapsed();
      ck(cuGraphicsUnmapResources(1, &slot.resource, copy_stream)); mapped=false;
      trace.unmap_ms=elapsed();
      ck(cuEventRecord(copy_end, copy_stream));
      ck(cuStreamSynchronize(copy_stream));
      trace.synchronize_ms=elapsed();
      float gpu_span=0; ck(cuEventElapsedTime(&gpu_span, copy_begin, copy_end));
      trace.gpu_span_ms=gpu_span;
      trace.total_ms=seconds(Clock::now()-start)*1000;
      // Caller may release its tensor now; the queued D3D texture owns pixels.
      std::lock_guard<std::mutex> lock(mutex);
      // Keep presenting the old queue while the first replacement is prepared.
      // Clearing before the GPU copy completed created avoidable empty deadlines
      // during scene changes. Publication and retirement are one transaction.
      if (replace) {
        stats.discarded+=ready.size();
        while (!ready.empty()) { free.push_back(ready.front()); ready.pop_front(); }
      }
      slot.token=token; ready.push_back(index); ++stats.submitted;
      stats.copy_ms=seconds(Clock::now()-start)*1000;
      copy_trace=trace;
      changed.notify_all(); return 1;
    } catch (const std::exception& e) {
      Context context(cuda);
      if (mapped) cuGraphicsUnmapResources(1, &slot.resource, copy_stream);
      cuStreamSynchronize(copy_stream);
      std::lock_guard<std::mutex> lock(mutex);
      free.push_back(index); error=e.what(); changed.notify_all(); return -1;
    }
  }
  void clear() {
    std::lock_guard<std::mutex> lock(mutex);
    stats.discarded+=ready.size();
    while (!ready.empty()) { free.push_back(ready.front()); ready.pop_front(); }
    changed.notify_all();
  }
  bool wait(Clock::time_point deadline) {
    auto remaining=deadline-Clock::now();
    if (remaining<=Clock::duration::zero()) return true;
    LARGE_INTEGER due{};
    due.QuadPart=-std::max<LONGLONG>(1, (LONGLONG)(seconds(remaining)*1e7));
    if (!SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0))
      throw std::runtime_error("Could not arm presentation timer");
    HANDLE events[]={stop_event,timer};
    return WaitForMultipleObjects(2,events,FALSE,INFINITE)==WAIT_OBJECT_0+1;
  }
  void run() {
    DWORD task=0;
    HANDLE mmcss=AvSetMmThreadCharacteristicsW(L"Playback", &task);
    if (mmcss) AvSetMmThreadPriority(mmcss, AVRT_PRIORITY_HIGH);
    else SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    const auto period=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1/fps));
    auto deadline=Clock::now(), previous=deadline;
    bool primed=false, starved=false, have_previous=false;
    try {
      for (;;) {
        {
          std::unique_lock<std::mutex> lock(mutex);
          if (!primed) {
            changed.wait(lock,[&]{return stopping || !error.empty() || ready.size()>=prime;});
            if (stopping || !error.empty()) break;
            deadline=Clock::now(); primed=true;
          }
        }
        if (!wait(deadline)) break;
        size_t index;
        double empty_wait_ms=0;
        uint64_t queued_before=0;
        {
          std::unique_lock<std::mutex> lock(mutex);
          if (stopping || !error.empty()) break;
          // Empty before the deadline is normal. Only count starvation once
          // the next frame is due, and never reset the clock after a short wait.
          if (ready.empty()) {
            if (!starved) { ++stats.underruns; starved=true; }
            auto waiting=Clock::now();
            changed.wait(lock,[&]{return stopping || !error.empty() || !ready.empty();});
            empty_wait_ms=seconds(Clock::now()-waiting)*1000;
            if (stopping || !error.empty()) break;
          }
          queued_before=ready.size();
          index=ready.front(); ready.pop_front();
        }
        auto began=Clock::now();
        long before=spout.GetFrame();
        if (!spout.SendTexture(slots[index].texture)) throw std::runtime_error("Spout SendTexture failed");
        const long published_frame=spout.GetFrame();
        const bool advanced=published_frame!=before;
        auto done=Clock::now();
        {
          std::lock_guard<std::mutex> lock(mutex);
          const double gap_ms=have_previous ? seconds(done-previous)*1000 : 0;
          const double late_ms=std::max(0.,seconds(began-deadline)*1000);
          const double send_ms=seconds(done-began)*1000;
          // Record only a real starvation or substantially delayed handoff.
          // A deadline miss counter alone cannot distinguish a sub-ms wait
          // from a visible freeze. No extra GPU queries, waits or file I/O.
          if (empty_wait_ms>0 || gap_ms>1250/fps || late_ms>500/fps ||
              send_ms>500/fps || !advanced) {
            PresentSample sample;
            sample.token=slots[index].token; sample.at=wall();
            sample.gap_ms=gap_ms; sample.late_ms=late_ms;
            sample.empty_wait_ms=empty_wait_ms; sample.send_ms=send_ms;
            sample.queued_before=queued_before; sample.advanced=advanced ? 1 : 0;
            present_trace.add(sample);
          }
          if (advanced) {
            ++stats.sent; stats.last_token=slots[index].token;
            stats.last_send_at=wall(); if (!stats.first_send_at) stats.first_send_at=stats.last_send_at;
            FrameSample frame;
            frame.token=slots[index].token; frame.sender_frame=published_frame;
            frame.published_ms=stats.last_send_at*1000;
            frame.monotonic_ms=seconds(done.time_since_epoch())*1000;
            frame_trace.add(frame);
            if (have_previous) {
              double gap=seconds(done-previous)*1000;
              stats.max_gap_ms=std::max(stats.max_gap_ms,gap);
              stats.gap_sum_ms+=gap; stats.gap_square_sum+=gap*gap;
            }
            previous=done; have_previous=true; starved=false;
          } else ++stats.access_misses;
          stats.send_ms=seconds(done-began)*1000;
          stats.late_ms=std::max(0.,seconds(began-deadline)*1000);
          free.push_back(index); changed.notify_all();
          deadline+=period;
          // A late wake never emits a catch-up burst or permanently slows the clock.
          while (deadline<=done) { deadline+=period; ++stats.missed_ticks; }
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(mutex); error=e.what(); changed.notify_all();
    }
    if (mmcss) AvRevertMmThreadCharacteristics(mmcss);
  }
};
static thread_local std::string last_error;
#define API extern "C" __declspec(dllexport)
API void* jsp_create(const char* name,unsigned w,unsigned h,double fps,unsigned capacity,unsigned prime) {
  try { return new Sender(name,w,h,fps,capacity,prime); }
  catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
API int jsp_submit(Sender* p,uint64_t pixels,uint64_t event,uint64_t token) {
  try { return p->submit((CUdeviceptr)pixels,(CUevent)(uintptr_t)event,token); }
  catch (const std::exception& e) { last_error=e.what(); return -1; }
}
API int jsp_replace(Sender* p,uint64_t pixels,uint64_t event,uint64_t token) {
  try { return p->submit((CUdeviceptr)pixels,(CUevent)(uintptr_t)event,token,true); }
  catch (const std::exception& e) { last_error=e.what(); return -1; }
}
API void jsp_clear(Sender* p) { p->clear(); }
API void jsp_stats(Sender* p,Stats* out) {
  std::lock_guard<std::mutex> lock(p->mutex); *out=p->stats; out->queued=p->ready.size();
}
API int jsp_copy_trace(Sender* p,CopyTrace* out,unsigned bytes) {
  if (!p || !out || bytes!=sizeof(CopyTrace)) return 0;
  std::lock_guard<std::mutex> lock(p->mutex); *out=p->copy_trace; return 1;
}
API int jsp_present_trace(Sender* p,PresentSample* out,unsigned capacity,unsigned bytes) {
  if (!p) return -1;
  std::lock_guard<std::mutex> lock(p->mutex);
  return p->present_trace.copy(out,capacity,bytes);
}
API int jsp_frame_trace(Sender* p,FrameSample* out,unsigned capacity,unsigned bytes) {
  if (!p) return -1;
  std::lock_guard<std::mutex> lock(p->mutex);
  return p->frame_trace.copy(out,capacity,bytes);
}
API int jsp_set_copy_priority(Sender* p,int priority) {
  if (!p) return 0;
  try { p->set_copy_priority(priority); return 1; }
  catch (const std::exception& e) { last_error=e.what(); return -1; }
}
API const char* jsp_error(Sender* p) {
  if (p) { std::lock_guard<std::mutex> lock(p->mutex); last_error=p->error; }
  return last_error.c_str();
}
API void jsp_close(Sender* p) { delete p; }
