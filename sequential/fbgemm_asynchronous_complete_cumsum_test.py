import torch
from sequential.fbgemm import (
    asynchronous_complete_cumsum,
) 

t_in_example = torch.tensor([7, 8, 2, 1, 0, 9, 4], dtype=torch.int64)
res1 = asynchronous_complete_cumsum(t_in_example)
print(res1)

t_in_empty = torch.tensor([], dtype=torch.float32)
res2 = asynchronous_complete_cumsum(t_in_empty)
print(res2)

t_in_scalar = torch.tensor(5, dtype=torch.int32)
res3 =  asynchronous_complete_cumsum(t_in_scalar)
print(res3)
