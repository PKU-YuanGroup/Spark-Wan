import numpy as np
import torch.distributed as dist


# Global variables
sequence_parrallel_group = None


def get_rank():
    return dist.get_rank()


def get_sequence_parallel_group():
    return sequence_parrallel_group


def init_sequence_parallel_group(sp_size: int):
    if sp_size == 1:
        return

    global sequence_parrallel_group
    # Get global rank
    global_rank = get_rank()
    dist.barrier()

    # Get sequence group index and local rank in sequence group
    sp_group_index = global_rank // sp_size
    sp_group_local_rank = global_rank % sp_size

    dp_size = dist.get_world_size() // sp_size

    for i in range(dp_size):
        sp_group_rank_start = i * sp_size
        sp_group_rank_end = (i + 1) * sp_size
        sp_group_ranks = np.arange(sp_group_rank_start, sp_group_rank_end)
        sp_group = dist.new_group(ranks=sp_group_ranks)
        if global_rank in sp_group_ranks:
            sequence_parrallel_group = sp_group

    dist.barrier()
    return sp_group_index, sp_group_local_rank


def setup_sequence_parallel_group(sp_size: int):
    global_rank = get_rank()
    dp_size = dist.get_world_size()
    sp_size = sp_size
    if sp_size > 1:
        dp_size = dp_size // sp_size
    dp_rank = global_rank // sp_size

    if sp_size > 1:
        sp_group_index, sp_group_local_rank = init_sequence_parallel_group(sp_size)
    else:
        sp_group_index = 0
        sp_group_local_rank = 0

    return sp_group_index, sp_group_local_rank, dp_rank, dp_size
