import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr, x_row_stride, out_row_stride, D, BLOCK: tl.constexpr
):

    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    x = tl.load(
        x_ptr + pid * x_row_stride + offsets, mask=offsets < D, other=-float("inf")
    )

    x_max = tl.max(x, axis=0)
    x_minus_max = x - x_max

    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator

    tl.store(out_ptr + pid * out_row_stride + offsets, out, mask=offsets < D)


def softmax(x):
    assert x.ndim == 2

    out = torch.empty_like(x)

    n, d = x.shape
    BLOCK = triton.next_power_of_2(d)
    if BLOCK <= 2048:
        num_warps = 4
    elif BLOCK <= 8192:
        num_warps = 8
    elif BLOCK <= 16384:
        num_warps = 16
    else:
        num_warps = 32
    grid = (n,)

    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), d, BLOCK=BLOCK, num_warps=num_warps
    )

    return out


if __name__ == "__main__":
    from triton.testing import do_bench

    def naive_softmax(x):
        x_max = x.max(dim=1, keepdim=True).values
        x_minus_max = x - x_max

        numerator = torch.exp(x_minus_max)
        denominator = numerator.sum(dim=1, keepdim=True)
        out = numerator / denominator

        return out

    # Benchmark over triton, compile and eager
    n = 4096

    naive = lambda: naive_softmax(x)
    eager = lambda: torch.softmax(x, dim=-1)
    yours = lambda: softmax(x)

    torch.testing.assert_close(yours(), eager())

    print(f"{'d':8s} {'naive':14s} {'eager':14s} {'torch.compile':14s} {'triton':14s}")
    for d in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        compiled_fn = torch.compile(lambda x: torch.softmax(x, dim=-1), dynamic=False)
        compiled = lambda: compiled_fn(x)

        x = torch.randn(n, d).cuda()
        ms_naive = do_bench(naive, warmup=50, rep=1000)
        ms_eager = do_bench(eager, warmup=50, rep=1000)
        ms_compiled = do_bench(compiled, warmup=50, rep=1000)
        ms_triton = do_bench(yours, warmup=50, rep=1000)
        print(
            f"{d:<8d} {ms_naive:<14.3f} {ms_eager:<14.3f} {ms_compiled:<14.3f} {ms_triton:<14.3f}"
        )
