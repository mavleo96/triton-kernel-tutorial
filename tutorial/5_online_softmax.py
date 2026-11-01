import torch
import triton
import triton.language as tl
from utils import bench_sweep


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr, stride_xn, stride_on, D, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)

    m = -float("inf")
    s = 0.0
    for col_start in tl.range(0, D, BLOCK_D):
        col_offsets = col_start + tl.arange(0, BLOCK_D)

        x = tl.load(
            x_ptr + pid * stride_xn + col_offsets,
            mask=col_offsets < D,
            other=-float("inf"),
        )

        new_m = tl.maximum(m, tl.max(x, axis=0))
        x_minus_max = x - new_m
        exp_x = tl.exp(x_minus_max)
        s = s * tl.exp(m - new_m) + tl.sum(exp_x, axis=0)
        m = new_m

    for col_start in tl.range(0, D, BLOCK_D):
        col_offsets = col_start + tl.arange(0, BLOCK_D)

        x = tl.load(
            x_ptr + pid * stride_xn + col_offsets,
            mask=col_offsets < D,
            other=-float("inf"),
        )

        out = tl.exp(x - m) / s
        tl.store(out_ptr + pid * stride_on + col_offsets, out, mask=col_offsets < D)


def online_softmax(x):
    assert x.ndim == 2
    assert x.stride(-1) == 1

    out = torch.empty_like(x)

    n, d = x.shape
    BLOCK_D = 4096
    num_warps = 8
    grid = (n,)

    online_softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), d, BLOCK_D=BLOCK_D, num_warps=num_warps
    )

    return out


if __name__ == "__main__":
    x_test = torch.randn(128, 4096, device="cuda")

    torch.testing.assert_close(online_softmax(x_test), torch.softmax(x_test, dim=-1))
    print("correct!")

    def naive_softmax(x: torch.Tensor):
        x_max = x.max(dim=1, keepdim=True).values
        x_minus_max = x - x_max

        numerator = torch.exp(x_minus_max)
        denominator = numerator.sum(dim=1, keepdim=True)
        out = numerator / denominator

        return out

    torch._dynamo.config.recompile_limit = 64
    eager_softmax = lambda x: torch.softmax(x, dim=-1)
    compiled = torch.compile(eager_softmax, dynamic=False)

    bench_sweep(
        {
            "triton": online_softmax,
            "naive": naive_softmax,
            "eager": eager_softmax,
            "torch.compile": compiled,
        },
        [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        lambda d: (torch.randn(128, d, device="cuda"),),
        xlabel="d",
        title="Compare: Triton vs. naive vs. eager vs. torch.compile",
        filename="online_softmax",
    )
