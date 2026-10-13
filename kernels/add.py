import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, x_stride, y_ptr, y_stride, out_ptr, out_stride, n, BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    x = tl.load(x_ptr + offsets * x_stride, mask=mask)
    y = tl.load(y_ptr + offsets * y_stride, mask=mask)

    tl.store(out_ptr + offsets * out_stride, x + y, mask=mask)


def add(x, y):
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape

    out = torch.empty_like(x)

    n = x.numel()
    BLOCK = 256
    # grid = lambda META: (triton.cdiv(n, META."BLOCK"),)
    grid = (triton.cdiv(n, BLOCK),)

    add_kernel[grid](x, x.stride(0), y, y.stride(0), out, out.stride(0), n, BLOCK)

    return out


if __name__ == "__main__":
    x = torch.arange(100, device="cuda")[::2]
    y = torch.arange(50, device="cuda")

    torch.testing.assert_close(add(x, y), x + y)
    print("correct!")
