#pragma once
#include <array>
#include <cstdint>
#include <type_traits>

// Token is Python's immutable pixel identity; sender_frame is Spout's actual
// public counter read by Easel. Do not infer one from the other after a drop.
struct FrameSample {
    uint64_t serial=0, token=0;
    int64_t sender_frame=0;
    double published_ms=0, monotonic_ms=0;
};
static_assert(std::is_standard_layout<FrameSample>::value, "C ABI");
static_assert(sizeof(FrameSample)==40, "Explicit diagnostic ABI");

class FrameTrace {
public:
    static constexpr unsigned capacity=256;
    void add(FrameSample sample) {
        sample.serial=++m_serial;
        m_samples[(m_serial-1)%capacity]=sample;
    }
    int copy(FrameSample* out, unsigned limit, unsigned bytes) const {
        if (!out || !limit || bytes!=sizeof(FrameSample)) return -1;
        const unsigned count=static_cast<unsigned>(m_serial<capacity ? m_serial : capacity);
        const unsigned n=count<limit ? count : limit;
        for (unsigned i=0;i<n;++i) out[i]=m_samples[(m_serial-n+i)%capacity];
        return static_cast<int>(n);
    }
private:
    std::array<FrameSample,capacity> m_samples{};
    uint64_t m_serial=0;
};
