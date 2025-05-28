import torch
from sequential.fbgemm import (
    jagged_to_padded_dense,
) 

# Example from the docstring
values_example = torch.tensor([[1,1],[2,2],[3,3],[4,4]], dtype=torch.int64) # Explicit dtype for matching example
offsets_example_dim0 = torch.tensor([0, 1, 3])

# Call the implemented function
result_example = jagged_to_padded_dense(
    values_example,
    [offsets_example_dim0],  # offsets is a list of tensors
    [3],                     # max_lengths is a list of ints
    padding_value=7
)
print("Example output:")
print(result_example)
print("Expected output:")
print(torch.tensor([[[1, 1], [7, 7], [7, 7]], [[2, 2], [3, 3], [7, 7]]], dtype=torch.int64))

# Example with scalar values and multiple jagged dimensions
scalar_values = torch.arange(1, 8, dtype=torch.float32) # [1,2,3,4,5,6,7]
# Innermost: items are length 2, 1, 2, 2
offsets_inner = torch.tensor([0, 2, 3, 5, 7]) # Defines 4 intermediate items
max_len_inner = 2
# Outermost: 2 bags, first bag has 2 intermediate items, second has 2 intermediate items
offsets_outer = torch.tensor([0, 2, 4]) # Operates on the 4 intermediate items
max_len_outer = 2

# offsets list: [outermost, innermost]
multi_dim_offsets = [offsets_outer, offsets_inner]
multi_dim_max_lengths = [max_len_outer, max_len_inner]

result_multi_dim = jagged_to_padded_dense(
    scalar_values,
    multi_dim_offsets,
    multi_dim_max_lengths,
    padding_value=0.0
)
print("\nMulti-dimensional example output:")
print(result_multi_dim)
# Expected intermediate (after inner processing):
# item1: [1,2], item2: [3,0], item3: [4,5], item4: [6,7]
# -> torch.tensor([[1,2],[3,0],[4,5],[6,7]])
# Expected final (after outer processing):
# bag1: [item1, item2] -> [[1,2],[3,0]]
# bag2: [item3, item4] -> [[4,5],[6,7]]
# -> torch.tensor([[[1,2],[3,0]], [[4,5],[6,7]]])
print("Expected multi-dimensional output:")
print(torch.tensor([[[1., 2.], [3., 0.]], [[4., 5.], [6., 7.]]], dtype=torch.float32))