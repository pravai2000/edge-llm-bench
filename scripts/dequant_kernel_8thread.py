import time
import mlx.core as mx

body = """
    uint i = thread_position_in_grid.x;          // one thread per packed word
    uint32_t word = wq[i];
    float s = scales[i / 8];                     // 8 words = 64 weights = one group
    float b = biases[i / 8];
    for (uint k = 0; k < 8; k++) {
        out[i * 8 + k] = float((word >> (4 * k)) & 0xF) * s + b;
    }
"""
kernel = mx.fast.metal_kernel(name="dequant4", input_names=["wq", "scales", "biases"],
                              output_names=["out"], source=body)

w = mx.random.normal(shape=(4096, 4096))
wq, scales, biases = mx.quantize(w, group_size=64, bits=4)

def mine():
    return kernel(inputs=[wq, scales, biases], grid=(w.size // 8, 1, 1), threadgroup=(256, 1, 1),
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
