#include "PresentTrace.h"
#include <iostream>
#include <stdexcept>

static void require(bool pass,const char* reason) {
    if (!pass) throw std::runtime_error(reason);
}
int main() {
    try {
        PresentTrace history;
        PresentSample output[64]{};
        require(history.copy(output,64,sizeof(PresentSample))==0,"empty history");
        for (uint64_t i=1;i<=90;++i) {
            PresentSample sample;
            sample.token=i+100; sample.at=double(i); sample.empty_wait_ms=double(i)/2;
            sample.queued_before=1; sample.advanced=1;
            history.add(sample);
        }
        require(history.copy(output,64,sizeof(PresentSample))==64,"bounded history");
        require(output[0].serial==27 && output[63].serial==90,"chronological wrap");
        require(output[0].token==127 && output[63].empty_wait_ms==45,"values retained");
        require(history.copy(output,2,sizeof(PresentSample))==2,"bounded caller");
        require(output[0].serial==89 && output[1].serial==90,"newest entries first in range");
        output[0].serial=999;
        require(history.copy(output,64,sizeof(PresentSample)-8)==-1,"reject old ABI");
        require(output[0].serial==999,"no write on bad ABI");
        require(history.copy(nullptr,64,sizeof(PresentSample))==-1,"reject null output");
        require(history.copy(output,0,sizeof(PresentSample))==-1,"reject zero capacity");
        require(history.copy(output,1000,sizeof(PresentSample))==64,"clamp large request");
        std::cout << "11 presentation-history assertions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
