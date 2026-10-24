#pragma once
#include <array>
#include <cstdint>
#include <type_traits>

// Independent diagnostics ABI. Ownership and synchronization belong to Sender;
// this bounded history does not allocate or change presentation deadlines.
struct PresentSample {
    uint64_t serial=0, token=0;
    double at=0, gap_ms=0, late_ms=0, empty_wait_ms=0, send_ms=0;
    uint64_t queued_before=0, advanced=0;
};
static_assert(std::is_standard_layout<PresentSample>::value, "C ABI requires standard layout");
static_assert(sizeof(PresentSample)==72, "Keep diagnostic ABI explicit");

class PresentTrace {
public:
    static constexpr unsigned capacity=64;
    void add(PresentSample sample) {
        sample.serial=++m_serial;
        m_samples[(m_serial-1)%capacity]=sample;
    }
    // Return newest samples, oldest first. Refuse a mismatched caller layout
    // before touching its memory, including during mixed-version rollback.
    int copy(PresentSample* out, unsigned limit, unsigned bytes) const {
        if (!out || !limit || bytes!=sizeof(PresentSample)) return -1;
        const auto count=static_cast<unsigned>(
            m_serial<capacity ? m_serial : capacity);
        const unsigned n=count<limit ? count : limit;
        for (unsigned i=0;i<n;++i)
            out[i]=m_samples[(m_serial-n+i)%capacity];
        return static_cast<int>(n);
    }
private:
    std::array<PresentSample,capacity> m_samples{};
    uint64_t m_serial=0;
};
