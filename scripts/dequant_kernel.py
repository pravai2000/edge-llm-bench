import time
import mlx.core as mx

body = """
    uint idx = thread_position_in_grid.x;       // flat element index
    uint32_t word = wq[idx / 8];                // 8 four-bit values per uint32
    uint shift = (idx % 8) * 4;
    float q = float((word >> shift) & 0xF);
    uint g = idx / 64;                          // one scale/bias per 64 weights
    out[idx] = q * scales[g] + biases[g];
"""
kernel = mx.fast.metal_kernel(name="dequant4", input_names=["wq", "scales", "biases"],
                              output_names=["out"], source=body)

w = mx.random.normal(shape=(4096, 4096))
wq, scales, biases = mx.quantize(w, group_size=64, bits=4)

def mine():
    return kernel(inputs=[wq, scales, biases], grid=(w.size, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[w.shape], output_dtypes=[mx.float32])[0]

ref = mx.dequantize(wq, scales, biases, group_size=64, bits=4)
out = mine()
print("max abs diff vs mx.dequantize:", mx.abs(out - ref).max().item())

for name, f in (("mine", mine), ("mlx", lambda: mx.dequantize(wq, scales, biases, group_size=64, bits=4))):
    mx.eval(f())
    t = time.perf_counter()
    for _ in range(20):
        mx.eval(f())
    print(name, f"{(time.perf_counter() - t) / 20 * 1e3:.2f} ms")
