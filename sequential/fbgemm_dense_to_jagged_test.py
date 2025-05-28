import torch
from sequential.fbgemm import (
    dense_to_jagged,
) 

dense = torch.tensor([[[1, 1], [0, 0], [0, 0]], [[2, 2], [3, 3], [0, 0]]])
x_offsets = torch.tensor([0, 1, 3])
result_example = dense_to_jagged(dense, [x_offsets])


print("Example output:")
print(result_example)
print("Expected output:")
print((torch.tensor([[1,1], [2,2], [3,3]], dtype=torch.int64), [torch.tensor([0, 1, 3])]))
