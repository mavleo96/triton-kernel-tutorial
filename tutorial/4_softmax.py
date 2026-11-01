import torch
import triton
import triton.language as tl
from utils import bench_sweep


@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride_xn, stride_on, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)

    x = tl.load(
        x_ptr + pid * stride_xn + offsets, mask=offsets < D, other=-float("inf")
    )

    x_max = tl.max(x, axis=0)
    x_minus_max = x - x_max

    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator

    tl.store(out_ptr + pid * stride_on + offsets, out, mask=offsets < D)


def softmax(x):
    assert x.ndim == 2

    out = torch.empty_like(x)

    n, d = x.shape
    BLOCK_D = triton.next_power_of_2(d)
    if BLOCK_D <= 2048:
        num_warps = 4
    elif BLOCK_D <= 8192:
        num_warps = 8
    elif BLOCK_D <= 16384:
        num_warps = 16
    else:
        num_warps = 32
    grid = (n,)

    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), d, BLOCK_D=BLOCK_D, num_warps=num_warps
    )

    return out


if __name__ == "__main__":
    x_test = torch.randn(4096, 1024, device="cuda")

    torch.testing.assert_close(softmax(x_test), torch.softmax(x_test, dim=-1))
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
            "naive": naive_softmax,
            "eager": eager_softmax,
            "torch.compile": compiled,
            "triton": softmax,
        },
        [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        lambda d: (torch.randn(128, d, device="cuda"),),
        xlabel="d",
        title="Compare: Triton vs. torch.compile vs. eager",
        filename="softmax",
    )
