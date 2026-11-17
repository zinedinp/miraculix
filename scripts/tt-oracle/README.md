# tt-oracle

Reproduces the bit-exact checksums that `src/formats/bfp.rs` tests against.
It runs Tenstorrent's own `row_scalar` (tt-metal `tt_metal/impl/data_format/bfp_simd.hpp`,
Apache-2.0) over a fixed pseudo-random stream and prints an FNV-1a hash of the output.

```sh
curl -sSLO https://raw.githubusercontent.com/tenstorrent/tt-metal/main/tt_metal/impl/data_format/bfp_simd.hpp
g++ -std=c++17 -O1 -Istub -I. oracle.cpp -o oracle && ./oracle
```

`stub/` replaces `tt-metalium/bfloat16.hpp` with a bare `bfloat16` struct. The input
generator in `oracle.cpp` is mirrored in `src/formats/tests.rs` (`bfp_oracle_*`).
