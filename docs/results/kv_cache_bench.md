# KV-cache speedup benchmark

Config: `bench-tiny` (n_layer=2, d_model=32, max_seq=64), device=`cpu`.

Differentiation move #2: shows the cost of re-computing K/V on every step (no-cache) versus the gpt-fast pattern (static cache, indexed writes). Numbers are local; production T4 numbers land in Phase F.

| ctx | no-cache tok/s | cached tok/s | speedup |
|----:|---------------:|-------------:|--------:|
| 32 | 1882.4 | 2331.5 | 1.24x |
| 64 | 2067.5 | 2443.5 | 1.18x |
