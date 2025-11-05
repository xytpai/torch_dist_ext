import os
import torch
import torch.distributed as dist


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)


CommWorkspace = eval(f"torch.classes.{package_name}.CommWorkspace")
allreduce_rms_fusion = eval(f"torch.ops.{package_name}.allreduce_rms_fusion")


class CommProcess:
    def __init__(self, rank, world_size, size_in_bytes):
        torch.cuda.set_device(rank)
        nblocks = 256
        self.comm = CommWorkspace(rank, world_size, nblocks, size_in_bytes)
        handle = self.comm.get_handle()
        handle_list = [None] * world_size
        dist.all_gather_object(handle_list, handle)
        dist.barrier()
        self.comm.open_handles(handle_list)
        dist.barrier()

    def workspace(self):
        return self.comm.get_workspace()
