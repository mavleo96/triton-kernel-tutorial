import torch
import triton
import triton.language as tl
from utils import bench_sweep


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


def relu_fused(x, w, b, BLOCK=256):
    out = torch.empty_like(x)

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
    x_test = torch.randn(1000, 512, device="cuda")
    w_test = torch.randn(512, device="cuda")
    b_test = torch.randn(512, device="cuda")

    eager_func = lambda x, w, b: torch.relu(x * w + b)

    torch.testing.assert_close(
        eager_func(x_test, w_test, b_test), relu_fused(x_test, w_test, b_test)
    )
    print("correct!")

    # Benchmark over triton, compile and eager
    # dynamic=False recompiles a shape-specialized kernel for every n; raise the
    # recompile limit (default 8) so later sizes don't silently fall back to eager.
    torch._dynamo.config.recompile_limit = 64
    compiled = torch.compile(eager_func, dynamic=False)
    bench_sweep(
        {"eager": eager_func, "torch.compile": compiled, "triton": relu_fused},
        [2**i for i in range(7, 16)],
        lambda n: (
            torch.randn(n, 512, device="cuda"),
            torch.randn(512, device="cuda"),
            torch.randn(512, device="cuda"),
        ),
        name="relu_fused",
        xlabel="n",
    )

    # Benchmark over block size
    x = torch.randn(100, 1024, device="cuda")
    w = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    bench_sweep(
        {"triton": relu_fused},
        [128, 256, 512, 1024, 2048, 4096],
        lambda block: (x, w, b, block),
        name="relu_fused_block",
        xlabel="BLOCK",
    )
