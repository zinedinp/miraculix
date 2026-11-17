// Oracle for src/formats/gguf.rs: llama.cpp's dequantize_row_q8_0 / dequantize_row_q6_K,
// copied verbatim from ggml/src/ggml-quants.c (MIT) between the markers, with minimal type stubs.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define GGML_RESTRICT __restrict
#define QK8_0 32
#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;
static float GGML_FP16_TO_FP32(ggml_half h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }
// ---- verbatim ----
void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}
void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
// ---- end verbatim ----
static uint64_t S = 0x9E3779B97F4A7C15ull, H = 0xcbf29ce484222325ull;
static uint64_t nx(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return S; }
static void hb(uint8_t b) { H ^= b; H *= 0x100000001b3ull; }
static void hf(float f) { uint32_t u; memcpy(&u, &f, 4); for (int i = 0; i < 4; i++) hb(u >> (8 * i)); }
static uint16_t rhalf(void) { // finite halves with moderate exponents
    uint64_t u = nx(); return (uint16_t)(((u >> 20) & 0x8000) | ((8 + (u >> 5) % 14) << 10) | (u & 0x3FF));
}
int main(void) {
    block_q8_0 b8[64]; float y8[64 * 32];
    for (int i = 0; i < 64; i++) { b8[i].d = rhalf(); for (int j = 0; j < 32; j++) b8[i].qs[j] = (int8_t)nx(); }
    dequantize_row_q8_0(b8, y8, 64 * 32);
    for (int i = 0; i < 64 * 32; i++) hf(y8[i]);
    printf("q8_0 0x%016llxull\n", (unsigned long long)H);
    H = 0xcbf29ce484222325ull;
    block_q6_K b6[16]; float y6[16 * 256];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 128; j++) b6[i].ql[j] = (uint8_t)nx();
        for (int j = 0; j < 64; j++) b6[i].qh[j] = (uint8_t)nx();
        for (int j = 0; j < 16; j++) b6[i].scales[j] = (int8_t)nx();
        b6[i].d = rhalf();
    }
    dequantize_row_q6_K(b6, y6, 16 * 256);
    for (int i = 0; i < 16 * 256; i++) hf(y6[i]);
    printf("q6_k 0x%016llxull\n", (unsigned long long)H);
}
