#include "FrameTrace.h"
#include <stdexcept>
#include <iostream>
static void require(bool value) { if (!value) throw std::runtime_error("Frame trace contract failed"); }
int main() {
    FrameTrace trace;
    FrameSample rows[256];
    require(trace.copy(rows,256,sizeof(FrameSample))==0);
    for (uint64_t i=0;i<600;++i) {
        FrameSample s;
        s.token=3*i; s.sender_frame=static_cast<int64_t>(i+10);
        s.published_ms=10000.+i*42.; s.monotonic_ms=2000.+i*42.;
        trace.add(s);
    }
    require(trace.copy(rows,256,sizeof(FrameSample))==256);
    require(rows[0].serial==345 && rows[255].serial==600);
    require(rows[0].token==1032 && rows[0].sender_frame==354);
    require(rows[255].published_ms==35158.);
    rows[0].token=12345;
    require(trace.copy(rows,5,sizeof(FrameSample))==5 && rows[0].serial==596);
    require(trace.copy(rows,1,sizeof(FrameSample)-1)==-1);
    require(trace.copy(nullptr,1,sizeof(FrameSample))==-1);
    require(trace.copy(rows,0,sizeof(FrameSample))==-1);
    std::cout << "Frame identity, bounded retention, copy and ABI checks passed\n";
}
