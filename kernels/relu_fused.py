import torch
import triton
import triton.language as tl


@triton.jit
def relu_fused_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    b_ptr,
    out_ptr,
    out_stride,
    n,
    d,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid1 = tl.program_id(0)
    pid2 = tl.program_id(1)

    n_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < n
    d_offsets = pid2 * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d

    x = tl.load(
        x_ptr + n_offsets[:, None] * x_stride[0] + d_offsets * x_stride[1],
        mask=n_mask[:, None] & d_mask,
    )

    w = tl.load(w_ptr + d_offsets, mask=d_mask)
    b = tl.load(b_ptr + d_offsets, mask=d_mask)

    y = x * w + b
    y_relu = tl.maximum(y, 0)

    tl.store(
        out_ptr + n_offsets[:, None] * out_stride[0] + d_offsets * out_stride[1],
        y_relu,
        mask=n_mask[:, None] & d_mask,
    )


def relu_fused(x, w, b):
    out = torch.empty_like(x)

    BLOCK = 256
    n, d = x.shape

    if triton.next_power_of_2(d) > BLOCK:
        BLOCK_D = BLOCK
        BLOCK_N = 1
        grid = (n, triton.cdiv(d, BLOCK))
    else:
        BLOCK_D = triton.next_power_of_2(d)
        BLOCK_N = BLOCK // BLOCK_D
        grid = (triton.cdiv(n, BLOCK_N), 1)

    relu_fused_kernel[grid](
        x, x.stride(), w, b, out, out.stride(), n, d, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )

    return out


if __name__ == "__main__":
    from triton.testing import do_bench

    # Benchmark over triton, compile and eager
    n = 50_000
    x = torch.randn(512, n).cuda().T
    w = torch.randn(512).cuda()
    b = torch.randn(512).cuda()

    eager = lambda: torch.relu(x * w + b)
    compiled_fn = torch.compile(lambda x, w, b: torch.relu(x * w + b))
    compiled = lambda: compiled_fn(x, w, b)
    yours = lambda: relu_fused(x, w, b)

    torch.testing.assert_close(yours(), eager())

    for name, f in [("eager", eager), ("torch.compile", compiled), ("triton", yours)]:
        ms = do_bench(f, warmup=50, rep=1000)
        print(f"{name:14s} {ms:.3f} ms")

    # Benchmark over block size
    n = 100
    x = torch.randn(n, 1024).cuda()
    w = torch.randn(1024).cuda()
    b = torch.randn(1024).cuda()

    for block_size in [128, 256, 512, 1024, 2048, 4096]:
        yours = lambda: relu_fused(x, w, b, BLOCK=block_size)
        ms = do_bench(yours, warmup=50, rep=1000)
        print(f"{block_size:14d} {ms:.3f} ms")
