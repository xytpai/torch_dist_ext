import os
import torch
import torch.distributed as dist
from typing import Tuple


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)


CommWorkspace = eval(f"torch.classes.{package_name}.CommWorkspace")
allreduce_rms_fusion_impl = eval(f"torch.ops.{package_name}.allreduce_rms_fusion")


class CommProcess:
    def __init__(self, rank, world_size, size_in_bytes, group):
        torch.cuda.set_device(rank)
        nblocks = 256
        self.comm = CommWorkspace(rank, world_size, nblocks, size_in_bytes)
        handle = self.comm.get_handle()
        handle_list = [None] * world_size
        dist.all_gather_object(handle_list, handle, group=group)
        # dist.barrier()
        self.comm.open_handles(handle_list)
        # dist.barrier()

    def workspace(self):
        return self.comm.get_workspace()

comm = None
workspace = None
def setup_env(rank, world_size, group=None):
    global comm
    global workspace
    comm = CommProcess(rank, world_size, 1024 * 1024 * 256, group)
    workspace = comm.workspace()
    workspace = workspace.cuda(rank)


@torch.library.custom_op("torch_dist_ext::get_workspace", mutates_args=[])
def get_workspace() -> torch.Tensor:
    assert workspace is not None
    return workspace

@torch.library.register_fake("torch_dist_ext::get_workspace")
def get_workspace_fake() -> torch.Tensor:
    return torch.empty(88, dtype=torch.uint8)

@torch.library.custom_op("torch_dist_ext::allreduce_rms_fusion_", mutates_args=[])
def allreduce_rms_fusion(rank: int, nranks: int, allreduce_in: torch.Tensor, 
    residual_in: torch.Tensor, rms_gamma: torch.Tensor, eps: float, workspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    size = allreduce_in.numel()
    hidden_dim = allreduce_in.shape[-1]
    if rank == 0:
        print(f"allreduce_in:{allreduce_in.shape}", flush=True)
    assert hidden_dim <= 8192
    norm_out = torch.empty_like(allreduce_in)
    residual_out = torch.empty_like(residual_in)
    allreduce_rms_fusion_impl(rank, nranks, allreduce_in, residual_in, rms_gamma, residual_out, norm_out, eps, workspace)
    return norm_out, residual_out

@torch.library.register_fake("torch_dist_ext::allreduce_rms_fusion_")
def allreduce_rms_fusion_fake(rank: int, nranks: int, allreduce_in: torch.Tensor, 
    residual_in: torch.Tensor, rms_gamma: torch.Tensor, eps: float, workspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_out = torch.empty_like(allreduce_in)
    residual_out = torch.empty_like(residual_in)
    return norm_out, residual_out

allreduce_rms_fusion_ = torch.ops.torch_dist_ext.allreduce_rms_fusion_

