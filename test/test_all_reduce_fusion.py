import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        rank=rank,
        world_size=world_size)


def worker(rank, world_size, allreduce_in, residual_in, rms_gamma, ref_norm_out, eps):
    setup(rank, world_size)
    num_tokens, hidden_dim = residual_in.shape
    local_allreduce_in = allreduce_in[rank].cuda(rank)
    local_residual_in = residual_in.cuda(rank)
    local_rms_gamma = rms_gamma.cuda(rank)
    dist.all_reduce(local_allreduce_in)
    local_norm_out = local_allreduce_in + local_residual_in
    local_norm_out = local_norm_out / ((local_norm_out**2).sum(dim=1, keepdim=True) / hidden_dim + eps).sqrt()
    local_norm_out *= local_rms_gamma
    maxdiff = (local_norm_out.cpu() - ref_norm_out).abs().max()
    print(f"rank:{rank}, maxdiff:{maxdiff}")
    dist.destroy_process_group()


def main():
    def testcase(world_size=8, num_tokens=128, hidden_dim=1024, eps=1e-6):
        allreduce_in = torch.randn(world_size, num_tokens, hidden_dim)
        residual_in = torch.randn(num_tokens, hidden_dim)
        rms_gamma = torch.randn(1, hidden_dim)
        ref_norm_out = allreduce_in.sum(dim=0) + residual_in
        ref_norm_out = ref_norm_out / ((ref_norm_out**2).sum(dim=1, keepdim=True) / hidden_dim + eps).sqrt()
        ref_norm_out *= rms_gamma
        mp.spawn(worker, args=(world_size, allreduce_in, residual_in, rms_gamma, ref_norm_out, eps), nprocs=world_size, join=True)
    testcase()


if __name__ == '__main__':
    main()
