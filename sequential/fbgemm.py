import torch

def _jagged_to_padded_dense_single_level(
    current_values: torch.Tensor,
    offsets_for_current_dim: torch.Tensor,
    max_len_for_current_dim: int,
    padding_value: float
) -> torch.Tensor:
    """
    Helper function to convert one level of jaggedness to a dense tensor.
    """
    # Validate offsets_for_current_dim dtype and basic structure
    if not (offsets_for_current_dim.ndim == 1 and \
            (offsets_for_current_dim.dtype == torch.int64 or \
             offsets_for_current_dim.dtype == torch.int32 or \
             offsets_for_current_dim.dtype == torch.long or \
             offsets_for_current_dim.dtype == torch.int)):
        raise TypeError(
            f"Offsets tensor must be a 1D tensor of integers (int32 or int64). "
            f"Got dtype: {offsets_for_current_dim.dtype}"
        )

    if offsets_for_current_dim.numel() > 0 and offsets_for_current_dim[0].item() != 0:
        raise ValueError("Offsets tensor must start with 0 if not empty.")

    # Determine number of bags/segments for the current jagged dimension
    if offsets_for_current_dim.numel() == 0:
        num_bags = 0
    elif offsets_for_current_dim.numel() == 1: # e.g., torch.tensor([0]) means 0 bags
        num_bags = 0
    else: # e.g., torch.tensor([0, N1, N2]) means 2 bags
        num_bags = len(offsets_for_current_dim) - 1
    
    if num_bags < 0: # Should be caught by above checks, but as a safeguard
        num_bags = 0

    # Determine the shape of the elements that are being grouped
    if current_values.ndim > 1:
        element_shape_tuple = current_values.shape[1:]
    else: # current_values is 1D (list of scalars) or 0D (a single scalar being wrapped)
        element_shape_tuple = ()

    # Define the output shape for this newly densified level
    # Shape: (num_bags, max_len_for_this_dim, *shape_of_elements)
    output_shape = (num_bags, max_len_for_current_dim) + element_shape_tuple
    
    if num_bags == 0:
        # If no bags are formed, return an empty tensor with the correct shape,
        # dtype, and device.
        return torch.empty(output_shape, dtype=current_values.dtype, device=current_values.device)

    # Create the output tensor, filled with the padding_value.
    # torch.full will handle casting padding_value to current_values.dtype.
    output_tensor = torch.full(
        output_shape,
        fill_value=padding_value,
        dtype=current_values.dtype,
        device=current_values.device
    )

    # Populate the output tensor with data from current_values
    for i in range(num_bags):
        start_idx = offsets_for_current_dim[i].item()
        end_idx = offsets_for_current_dim[i+1].item()

        # A segment is non-empty if start_idx < end_idx
        if start_idx < end_idx:
            # Slice the data for the current bag/segment from current_values
            bag_values_slice = current_values[start_idx:end_idx]
            actual_len = bag_values_slice.shape[0] # Number of items in this specific bag

            if actual_len > 0:
                # Determine how many items to copy (up to max_len_for_current_dim)
                len_to_copy = min(actual_len, max_len_for_current_dim)
                if len_to_copy > 0:
                    # Assign the sliced values to the output tensor.
                    # Ellipsis (...) correctly handles further dimensions of the elements.
                    if element_shape_tuple: # If elements are not scalars
                        output_tensor[i, :len_to_copy, ...] = bag_values_slice[:len_to_copy, ...]
                    else: # If elements are scalars
                        output_tensor[i, :len_to_copy] = bag_values_slice[:len_to_copy]
        # If start_idx >= end_idx, the segment is empty, and output_tensor[i] remains padding.
        
    return output_tensor


def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: list[torch.Tensor],
    max_lengths: list[int],
    padding_value: float = 0.0
) -> torch.Tensor:
    """
    Converts a jagged tensor into a dense tensor, padding with a specified padding value.

    Args:
        values (Tensor): Jagged tensor values. These are the innermost flat elements.
        offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.
                            The list is ordered from the outermost jagged dimension to the innermost.
                            For example, offsets[0] defines the segments for the first (outermost)
                            jagged dimension, and offsets[len(offsets)-1] defines segments for the
                            innermost jagged dimension (acting directly on `values`).
        max_lengths (int[]): A list with max_length for each jagged dimension.
                             max_lengths[k] is the maximum length for the k-th jagged dimension.
        padding_value (float): Value to set in the empty areas of the dense output,
                               outside of the jagged tensor coverage.

    Returns:
        Tensor: the padded dense tensor.

    Example:
        >>> values = torch.tensor([[1,1],[2,2],[3,3],[4,4]]) # Innermost elements
        >>> offsets_dim0 = torch.tensor([0, 1, 3]) # Defines 2 bags for the first jagged dimension
        >>> # The function expects a list of offset tensors
        >>> result = jagged_to_padded_dense(values, [offsets_dim0], [3], 7)
        >>> print(result)
        tensor([[[1, 1],
                 [7, 7],
                 [7, 7]],
        <BLANKLINE>
                [[2, 2],
                 [3, 3],
                 [7, 7]]], dtype=torch.int64)
    """
    # Input validation
    if not isinstance(values, torch.Tensor):
        raise TypeError("Input 'values' must be a torch.Tensor.")
    if not isinstance(offsets, list) or not all(isinstance(o, torch.Tensor) for o in offsets):
        raise TypeError("Input 'offsets' must be a list of torch.Tensors.")
    if not isinstance(max_lengths, list) or not all(isinstance(ml, int) for ml in max_lengths):
        raise TypeError("Input 'max_lengths' must be a list of ints.")
    if not isinstance(padding_value, (int, float)): # Allow int for padding_value as it's often convenient
        raise TypeError("Input 'padding_value' must be a number (int or float).")

    if len(offsets) != len(max_lengths):
        raise ValueError(
            "Inputs 'offsets' and 'max_lengths' must have the same number of elements, "
            "representing the same number of jagged dimensions."
        )

    if not offsets:
        # If the 'offsets' list is empty, it implies no jagged dimensions are specified.
        # In this case, the 'values' tensor is considered already dense.
        return values

    current_data = values
    num_jagged_dims = len(offsets)

    # Process jagged dimensions iteratively.
    # The `offsets` and `max_lengths` lists are defined from outermost to innermost dimension.
    # We need to apply the innermost transformation first (to `values`), then the next level, and so on.
    # So, we iterate from `offsets[num_jagged_dims - 1]` down to `offsets[0]`.
    for i in range(num_jagged_dims - 1, -1, -1):
        offsets_for_this_level = offsets[i]
        max_len_for_this_level = max_lengths[i]

        current_data = _jagged_to_padded_dense_single_level(
            current_values=current_data,
            offsets_for_current_dim=offsets_for_this_level,
            max_len_for_current_dim=max_len_for_this_level,
            padding_value=float(padding_value) # Helper expects float, consistent with docstring
        )
    
    return current_data


def dense_to_jagged(
    dense: torch.Tensor,
    x_offsets: list[torch.Tensor],
    total_L: int = None  # Optional
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Converts a dense tensor into a jagged tensor, given the desired offsets
    of the resulting jagged tensor.

    Args:
        dense (Tensor): A dense input tensor to be converted.
        x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension.
                              The list is ordered from outermost to innermost.
                              x_offsets[0] defines how to segment the first dimension of 'dense'.
                              x_offsets[k] defines how to segment the first dimension of the
                              tensor resulting from processing with x_offsets[k-1].
        total_L (int, Optional): Total number of values in the resulting jagged tensor's
                                 values component (i.e., `output_values.shape[0]`).
                                 If provided, used for validation.

    Returns:
        (Tensor, Tensor[]): A tuple containing:
                            - Tensor: The values of the resulting jagged tensor (a flattened tensor).
                            - Tensor[]: The offsets of the resulting jagged tensor
                                        (identical to the input `x_offsets`).

    Example:
        >>> dense_example = torch.tensor([[[1, 1], [0, 0], [0, 0]], \
                                          [[2, 2], [3, 3], [0, 0]]], dtype=torch.long)
        >>> x_offsets_example = torch.tensor([0, 1, 3], dtype=torch.long)
        >>> values, out_offsets = dense_to_jagged(dense_example, [x_offsets_example])
        >>> print(values)
        tensor([[1, 1],
                [2, 2],
                [3, 3]])
        >>> print(out_offsets)
        [tensor([0, 1, 3])]
    """

    # --- Input Validations ---
    if not isinstance(dense, torch.Tensor):
        raise TypeError("Input 'dense' must be a torch.Tensor.")
    if not isinstance(x_offsets, list) or \
       not all(isinstance(o, torch.Tensor) for o in x_offsets):
        raise TypeError("Input 'x_offsets' must be a list of torch.Tensors.")
    if total_L is not None and not isinstance(total_L, int):
        raise TypeError("Input 'total_L' must be an int or None.")

    if not x_offsets:
        # If x_offsets list is empty, 'dense' is considered the final 'values' tensor.
        # The returned offsets list will also be empty.
        if total_L is not None and dense.shape[0] != total_L:
            raise ValueError(
                f"When x_offsets is empty, 'dense' is returned as values. "
                f"dense.shape[0] ({dense.shape[0]}) must match total_L ({total_L})."
            )
        return dense, []

    current_tensor_to_segment = dense

    # --- Iterative Unrolling ---
    # Process each level of jaggedness defined by the tensors in x_offsets.
    # d_idx = 0 corresponds to the outermost jagged dimension.
    for d_idx in range(len(x_offsets)):
        offsets_for_current_level = x_offsets[d_idx]

        # Validate the current offsets tensor
        if not (offsets_for_current_level.ndim == 1 and
                (offsets_for_current_level.dtype == torch.int64 or
                 offsets_for_current_level.dtype == torch.int32 or
                 offsets_for_current_level.dtype == torch.long or  # Alias for int64
                 offsets_for_current_level.dtype == torch.int)):   # Alias for int32
            raise TypeError(
                f"Offsets tensor at index {d_idx} in x_offsets must be a 1D tensor of integers. "
                f"Got dtype: {offsets_for_current_level.dtype}."
            )
        if offsets_for_current_level.numel() > 0 and offsets_for_current_level[0].item() != 0:
            raise ValueError(
                f"Offsets tensor at index {d_idx} in x_offsets must start with 0 if not empty. "
                f"Got: {offsets_for_current_level}."
            )

        # Determine the number of segments ("bags") defined by this offset tensor
        num_segments_defined_by_offsets = 0
        if offsets_for_current_level.numel() > 1: # e.g., [0, N1, ...]
            num_segments_defined_by_offsets = len(offsets_for_current_level) - 1
        elif offsets_for_current_level.numel() == 1 and offsets_for_current_level[0].item() == 0: # e.g., [0]
            num_segments_defined_by_offsets = 0
        elif offsets_for_current_level.numel() == 0: # e.g., tensor([])
            num_segments_defined_by_offsets = 0
        # Else: malformed (e.g. [1] or non-empty but not starting with 0), caught by prior check.

        num_segments_in_input_tensor = current_tensor_to_segment.shape[0]

        # Check for compatibility between input tensor segments and offset-defined segments
        if num_segments_in_input_tensor != num_segments_defined_by_offsets:
            # Allow (0 segments in input, 0 segments defined by offsets) as a valid non-error case.
            if not (num_segments_in_input_tensor == 0 and num_segments_defined_by_offsets == 0):
                raise ValueError(
                    f"Dimension mismatch at jagged processing level {d_idx}: "
                    f"Input tensor (shape {current_tensor_to_segment.shape}) has "
                    f"{num_segments_in_input_tensor} segments (shape[0]), "
                    f"but offsets tensor {offsets_for_current_level} expects/defines "
                    f"{num_segments_defined_by_offsets} segments."
                )
        
        # If processing segments, the input tensor must have a "MaxLength" dimension (dim 1)
        if num_segments_defined_by_offsets > 0 and current_tensor_to_segment.ndim < 2:
            raise ValueError(
                f"Input tensor at level {d_idx} (shape {current_tensor_to_segment.shape}) "
                f"must have at least 2 dimensions (NumSegments, MaxLengthPerSegment, ...) "
                f"if offsets define more than 0 segments."
            )

        if num_segments_defined_by_offsets == 0:
            # If offsets define 0 segments, the result of this level's processing is an empty tensor.
            # The trailing dimensions should match the "element" shape from current_tensor_to_segment.
            element_trailing_dims = ()
            if current_tensor_to_segment.ndim >= 2: # (NumSegs, MaxLen, E1, E2...) -> El shape (E1, E2...)
                element_trailing_dims = current_tensor_to_segment.shape[2:]
            # If ndim is 1 (NumSegs,) or 0 (scalar), and num_segments_defined_by_offsets is 0,
            # the resulting element shape is also empty.
            
            empty_shape = (0,) + element_trailing_dims
            current_tensor_to_segment = torch.empty(
                empty_shape,
                dtype=current_tensor_to_segment.dtype,
                device=current_tensor_to_segment.device
            )
        else: # num_segments_defined_by_offsets > 0
            collected_value_segments = []
            # Max length available in each segment of the current dense tensor (from its dim 1)
            max_len_available_in_dense_segment = current_tensor_to_segment.shape[1]

            for i in range(num_segments_defined_by_offsets):
                # Length of the i-th segment in the *output* jagged structure for this level
                num_elements_to_take = (offsets_for_current_level[i+1].item() -
                                        offsets_for_current_level[i].item())

                if num_elements_to_take < 0:
                    raise ValueError(
                        f"Offsets at index {d_idx} for segment {i} imply negative length."
                    )
                
                if num_elements_to_take > max_len_available_in_dense_segment:
                    raise ValueError(
                        f"At jagged level {d_idx}, segment {i}: "
                        f"x_offsets demand {num_elements_to_take} elements, "
                        f"but dense segment has only {max_len_available_in_dense_segment} available "
                        f"(from shape[1] of tensor with shape {current_tensor_to_segment.shape})."
                    )
                
                # Extract data from the current dense segment
                # current_tensor_to_segment[i] is shape (max_len_available, ...further_dims)
                segment_data = current_tensor_to_segment[i, :num_elements_to_take]
                collected_value_segments.append(segment_data)
            
            # Concatenate all collected segments.
            # torch.cat handles a list of empty tensors correctly if all num_elements_to_take were 0.
            current_tensor_to_segment = torch.cat(collected_value_segments, dim=0)

    final_values = current_tensor_to_segment

    # --- Final Validations involving total_L ---
    if x_offsets: # This implies x_offsets is not empty and was processed.
        innermost_offsets_tensor = x_offsets[-1] # Innermost offsets determine final values length
        expected_L_from_innermost_offsets = 0
        if innermost_offsets_tensor.numel() > 0: # Not an empty tensor
            if innermost_offsets_tensor.numel() == 1 and innermost_offsets_tensor[0].item() == 0: # e.g. [0]
                 expected_L_from_innermost_offsets = 0
            elif innermost_offsets_tensor.numel() > 1:
                 expected_L_from_innermost_offsets = innermost_offsets_tensor[-1].item()
            # else: empty tensor implies 0, handled by init to 0
        
        if final_values.shape[0] != expected_L_from_innermost_offsets:
            raise RuntimeError(
                f"Internal inconsistency: final_values.shape[0] ({final_values.shape[0]}) "
                f"does not match length implied by innermost_offsets_tensor "
                f"({expected_L_from_innermost_offsets}). Innermost offsets: {innermost_offsets_tensor}."
            )
        
        if total_L is not None and total_L != expected_L_from_innermost_offsets:
            raise ValueError(
                f"Provided total_L ({total_L}) conflicts with length implied by "
                f"innermost_offsets_tensor ({expected_L_from_innermost_offsets})."
            )
    # If x_offsets was empty, total_L validation against 'dense' was done at the start.
            
    return final_values, x_offsets


def asynchronous_complete_cumsum(t_in: torch.Tensor) -> torch.Tensor:
    """
    Compute complete cumulative sum.
    For the purpose of this implementation, asynchronous behavior is not replicated.

    Args:
        t_in (Tensor): An input tensor.

    Returns:
        The complete cumulative sum of `t_in`. The shape of the output tensor
        is `t_in.numel() + 1`, and the first element is 0.

    Example:
        >>> t_in_example = torch.tensor([7, 8, 2, 1, 0, 9, 4], dtype=torch.int64)
        >>> asynchronous_complete_cumsum(t_in_example)
        tensor([ 0,  7, 15, 17, 18, 18, 27, 31])

        >>> t_in_empty = torch.tensor([], dtype=torch.float32)
        >>> asynchronous_complete_cumsum(t_in_empty)
        tensor([0.])

        >>> t_in_scalar = torch.tensor(5, dtype=torch.int32)
        >>> asynchronous_complete_cumsum(t_in_scalar)
        tensor([0, 5], dtype=torch.int32)
    """

    # Flatten the input tensor to treat it as a 1D sequence,
    # as implied by t_in.numel() for the output shape.
    t_in_flat = t_in.flatten()

    # Perform the standard cumulative sum.
    # PyTorch's torch.cumsum handles dtype promotion for small integer types
    # (e.g., int8, uint8, bool are promoted to int64) to prevent overflow.
    cumulative_sum_values = torch.cumsum(t_in_flat, dim=0)

    # Create the zero prefix.
    # It's good practice to use the input tensor's dtype for the zero prefix.
    # If torch.cumsum promotes the dtype (e.g., int8 -> int64),
    # torch.cat will handle the type promotion for the final result appropriately.
    zero_prefix = torch.tensor([0], dtype=t_in.dtype, device=t_in.device)

    # Concatenate the zero prefix with the computed cumulative sum.
    result = torch.cat((zero_prefix, cumulative_sum_values), dim=0)

    return result