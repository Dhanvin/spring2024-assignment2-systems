# Implements RmsNorm in triton
import torch
import triton 
import triton.language as tl

EPS = 1e-5

# The following function is shared by a 1D "grid" of programs, 
# where each program instance operates on one row in x (d_model).
#
# Assumptions (caller must verify):
#   - x and d are contiguous
#   - x is of shape [N, d_model], g is of shape [dmodel, ]
#   - BLOCK_SIZE > d_model.
#   - BLOCK_SIZE is a power of 2
#
@triton.jit
def rms_norm_fwd(
    x_ptr: tl.pointer_type,
    g_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    x_row_stride : tl.uint32, # needed for accessing corresponding row
    d_model : tl.uint32,
    BLOCK_SIZE: tl.constexpr):

    # Compute pointer offsets
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    output_row_start_ptr = output_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d_model

    # Load row from x and gain vector
    # NOTE: We must load in powers of 2
    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0)

    # Perform element-wise vector ops (similar notation as Numpy)
    x_row_sum_sq = tl.sqrt((tl.sum(x_row * x_row) / d_model) + EPS)
    x_normalized = (x_row * g) / x_row_sum_sq

    # Store output
    tl.store(output_row_start_ptr + offsets, x_normalized, mask=mask)

# Encapsulated torch.autograd.Function for inter-operation with Pytorch
class RmsNormTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gains):
        """
        Inputs:
            x: A tensor with shape (batch_size, seq_len, d_model)
            gains: A vector with shape (dmodel,)
        Returns:
            A tensor with same shape as x, normalized based on RMS along the last dim (d_model)
        """
        # TODO: Store stuff for backward pass

        d_model = x.shape[-1]
        assert x.is_contiguous()
        assert gains.is_contiguous()
        assert gains.shape == (d_model,)

        ctx.BLOCK_SIZE = triton.next_power_of_2(d_model)

        # Create a 2D view into x to compute stride.
        # -1 tells Pytorch to infer the size of that dimension automatically
        x_2d = x.view(-1, d_model)
        out_2d = torch.empty_like(x_2d)
        n_programs = x_2d.shape[0]
        row_stride = x_2d.stride(0)
        # breakpoint()

        # Create a 1D program-grid. Pass in tensors.
        # Internally, the triton.jit decorator will convert this
        # into a set of programs, and waits for all programs to complete before returning.
        rms_norm_fwd[(n_programs, )](x_2d, gains, out_2d, row_stride, d_model, ctx.BLOCK_SIZE)
        
        # Restore to shape of x and return
        return out_2d.view(x.shape)

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class RmsNormVanillaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gains):
        """
        Inputs:
            x: A tensor with shape (batch_size, seq_len, d_model)
            gains: A vector with shape (dmodel,)
        Returns:
            A tensor with same shape as x, normalized based on RMS along the last dim (d_model)
        """
        rms_normalization = (
                    x.pow(2).mean(dim=-1).add(EPS).rsqrt().unsqueeze(-1)
                )
        return x * rms_normalization * gains

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError