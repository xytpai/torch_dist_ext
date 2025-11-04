import torch
# print(dir(torch.ops.cgemm))
import cgemm
print(torch.ops.cgemm.allreduce_fusion)
# import cgemm

a=torch.rand(5).cuda()
b=torch.rand(5).cuda()
c = torch.ops.cgemm.allreduce_fusion(a,b)
print(a,b,c)