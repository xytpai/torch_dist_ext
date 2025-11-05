import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import cgemm


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps=1e-6, dtype=torch.float):
        super().__init__()
        self.eps = norm_eps
        self.weight = nn.Parameter(torch.randn(dim, dtype=dtype), requires_grad=False)

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(input_dtype)
        return self.weight * x


def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        rank=rank,
        world_size=world_size)


def worker(rank, world_size, allreduce_in, residual_in, rms, ref_norm_out, eps):
    setup(rank, world_size)
    num_tokens, hidden_dim = residual_in.shape
    local_allreduce_in = allreduce_in[rank].cuda(rank)
    local_residual_in = residual_in.cuda(rank)
    local_rms = rms.cuda(rank)
    dist.all_reduce(local_allreduce_in)    
    local_norm_out = local_rms(local_allreduce_in + local_residual_in)
    maxdiff = (local_norm_out.cpu() - ref_norm_out).abs().max()
    print(f"rank:{rank}, maxdiff:{maxdiff}")
    dist.destroy_process_group()


def main():
    def testcase(world_size=4, num_tokens=128, hidden_dim=1024, eps=1e-6):
        allreduce_in = torch.randn(world_size, num_tokens, hidden_dim)
        residual_in = torch.randn(num_tokens, hidden_dim)
        rms = RMSNorm(hidden_dim)
        ref_norm_out = rms(allreduce_in.sum(dim=0) + residual_in)
        mp.spawn(worker, args=(world_size, allreduce_in, residual_in, rms, ref_norm_out, eps), nprocs=world_size, join=True)
    testcase()


if __name__ == '__main__':
    main()
