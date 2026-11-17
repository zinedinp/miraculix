#include "bfp_simd.hpp"
#include <cstdio>
using namespace tt::tt_metal;
static uint64_t S = 0x9E3779B97F4A7C15ull;
static uint64_t nx() { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return S; }
static uint32_t gen_word(uint32_t row) {
    uint64_t u = nx();
    uint32_t kind = u & 7;
    if (kind == 0) return (uint32_t)(u >> 40) & 0x80000000u;
    if (kind == 1) return (uint32_t)(u >> 8) & 0x807FFFFFu;
    if (kind == 2 && row % 97 == 0) return (uint32_t)(u >> 8) | 0x7F800000u;
    uint32_t base = 120 + row % 20;
    uint32_t e = base + (uint32_t)((u >> 40) % 10) - 5;
    return ((uint32_t)(u >> 63) << 31) | (e << 23) | ((uint32_t)(u >> 8) & 0x7FFFFF);
}
static uint64_t H;
static void hb(uint8_t b) { H ^= b; H *= 0x100000001b3ull; }
template <int bits, bool bf> void run() {
    S = 0x9E3779B97F4A7C15ull; H = 0xcbf29ce484222325ull;
    for (uint32_t r = 0; r < 5000; ++r) {
        uint32_t w[16]; float f[16]; bfloat16 b[16];
        for (int j = 0; j < 16; ++j) { w[j] = gen_word(r); memcpy(&f[j], &w[j], 4); b[j].v = w[j] >> 16; }
        uint8_t e, out[16] = {0};
        if (bf) bfp_simd::row_scalar<bits, bfloat16>(b, &e, out); else bfp_simd::row_scalar<bits, float>(f, &e, out);
        hb(e);
        for (int k = 0; k < 16 * (bits + 1) / 8; ++k) hb(out[k]);
    }
    printf("bits=%d %s 0x%016llxull\n", bits, bf ? "bf16" : "f32", (unsigned long long)H);
}
int main() { run<7,false>(); run<7,true>(); run<3,false>(); run<3,true>(); run<1,false>(); run<1,true>(); }
