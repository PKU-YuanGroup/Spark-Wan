import torch
import torch.distributed as dist


def pad_tensor(tensor, pad_size, dim):
    return torch.cat(
        [
            tensor,
            torch.zeros(tensor.size(0), pad_size - tensor.size(dim), *tensor.shape[1:]),
        ],
        dim=dim,
    )


def unpad_tensor(tensor, unpad_size, dim):
    slice_indices = [slice(None)] * dim + [slice(None, unpad_size)]
    return tensor[tuple(slice_indices)]


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, x: torch.Tensor, gather_dim: int):
        if group is None:
            ctx.group = None
            return x

        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.group_local_rank = dist.get_rank(group)
        ctx.group_world_size = dist.get_world_size(group)

        output_tensor = [torch.empty_like(x) for _ in range(ctx.group_world_size)]
        dist.all_gather(output_tensor, x, group=group)
        return torch.cat(output_tensor, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.group:  # If not in sequence parallel group, return None
            return None, grad_output, None, None

        gather_dim_size = grad_output.size(ctx.gather_dim) // ctx.group_world_size
        split_grad = torch.split(grad_output, gather_dim_size, dim=ctx.gather_dim)[
            ctx.group_local_rank
        ].contiguous()
        split_grad *= ctx.group_world_size

        return (None, split_grad, None, None)


class SplitAndScatter(torch.autograd.Function):
    """Split tensor along a given dimension and scatter the result to different processes."""

    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, x: torch.Tensor, dim: int):
        if group is None:
            ctx.group = None
            return x

        ctx.group = group  # Save the group for backward
        ctx.dim = dim

        ctx.group_local_rank = dist.get_rank(group)
        ctx.group_world_size = dist.get_world_size(group)

        dim_size = x.size(dim)
        each_part_size = dim_size // ctx.group_world_size

        # Split the tensor and get the local part
        return torch.split(x, split_size_or_sections=each_part_size, dim=dim)[
            ctx.group_local_rank
        ].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.group:  # If not in sequence parallel group, return None
            return None, grad_output, None, None

        output = [torch.empty_like(grad_output) for _ in range(ctx.group_world_size)]
        dist.all_gather(output, grad_output, group=ctx.group)

        grad_output = torch.cat(output, dim=ctx.dim).contiguous()
        grad_output /= ctx.group_world_size

        return (None, grad_output, None, None)


class SplitAndAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, group: dist.ProcessGroup, x: torch.Tensor, split_dim: int, gather_dim: int
    ):
        if group is None:
            ctx.group = None
            return x

        ctx.group = group
        ctx.split_dim = split_dim
        ctx.gather_dim = gather_dim

        ctx.group_local_rank = dist.get_rank(group)
        ctx.group_world_size = dist.get_world_size(group)

        ctx.split_size = x.size(split_dim) // ctx.group_world_size
        ctx.gather_size = x.size(gather_dim) * ctx.group_world_size

        input_tensor = list(
            x.chunk(ctx.group_world_size, dim=split_dim)
        )  # b s/p d -> p b s/p d/p
        input_tensor = [
            x_chunk.contiguous() for x_chunk in input_tensor
        ]  # Important to ensure contiguous

        output_tensor = [
            torch.empty_like(input_tensor[0]) for _ in range(ctx.group_world_size)
        ]

        dist.all_to_all(output_tensor, input_tensor, group=group)

        x = torch.cat(output_tensor, dim=gather_dim).contiguous()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.group:  # If not in sequence parallel group, return None
            return None, grad_output, None, None

        group = ctx.group
        split_dim = ctx.split_dim
        gather_dim = ctx.gather_dim
        group_world_size = ctx.group_world_size

        input_tensor = list(grad_output.chunk(group_world_size, dim=gather_dim))
        input_tensor = [x.contiguous() for x in input_tensor]

        output_tensor = [
            torch.empty_like(input_tensor[0]) for _ in range(group_world_size)
        ]
        dist.all_to_all(output_tensor, input_tensor, group=group)
        grad_input = torch.cat(output_tensor, dim=split_dim).contiguous()
        return (None, grad_input, None, None)
