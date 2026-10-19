import torch
import triton
import triton.language as tl
from utils import bench_sweep

# Notes:
# 1. This is a simple add kernel that adds two vectors element-wise.
# 2. This demonstrates strides needs to be provided for non-contiguous tensors to work.


@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr, stride_x, stride_y, stride_out, n, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets * stride_x, mask=mask)
    y = tl.load(y_ptr + offsets * stride_y, mask=mask)

    tl.store(out_ptr + offsets * stride_out, x + y, mask=mask)


def add(x, y):
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape

    out = torch.empty_like(x)

    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    add_kernel[grid](
        x, y, out, x.stride(0), y.stride(0), out.stride(0), n, BLOCK_SIZE=BLOCK_SIZE
    )

    return out


if __name__ == "__main__":
    x_test = torch.arange(100, device="cuda")[::2]
    y_test = torch.arange(50, device="cuda")

    torch.testing.assert_close(add(x_test, y_test), x_test + y_test)
    print("correct!")

    bench_sweep(
        {"torch.add": torch.add, "triton": add},
        [2**i for i in range(7, 16)],
        lambda n: (torch.randn(n, device="cuda"), torch.randn(n, device="cuda")),
        xlabel="size",
        title="Compare: Triton vs. torch.add",
        filename="2_add",
    )
