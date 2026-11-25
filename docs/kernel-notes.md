the 0.0 difference and what it says about the packing order
both timings before and after the change, and the effective GB/s of each
one caution: real decode kernels never write dequantized weights to memory. They unpack inside the dot-product loop, so decode costs the read of the packed weights, not the write you measured here. That's why decode bandwidth ends up near 200 GB/s while this toy stays far below it.
