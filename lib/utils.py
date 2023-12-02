import torch.distributed as dist


def zero_rank_print(s):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
        print("########### " + s)
