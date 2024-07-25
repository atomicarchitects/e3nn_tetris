## Scatter mean function (courtesy of ChatGPT)

import torch

def scatter_mean(input, index=None, output_dim=None):
    if index is not None:
        # Case 1: Index is specified
        output_size = index.max().item() + 1
        output = torch.zeros(output_size, input.size(1), device=input.device)
        n = torch.zeros(output_size, device=input.device)
        
        for i in range(input.size(0)):
            idx = index[i]
            n[idx] += 1
            output[idx] += (input[i] - output[idx]) / n[idx]
        
        return output

    elif output_dim is not None:
        # Case 2: Index is skipped, output_dim is specified
        output = torch.zeros(len(output_dim), input.size(1), device=input.device)
        
        start_idx = 0
        for i, dim in enumerate(output_dim):
            end_idx = start_idx + dim
            if dim > 0:
                segment_sum = input[start_idx:end_idx].sum(dim=0)
                output[i] = segment_sum / dim
            start_idx = end_idx
        
        return output

    else:
        raise ValueError("Either 'index' or 'output_dim' must be specified.")

# # Example usage for Case 1 (index specified):
# input1 = torch.randn(3000, 144)
# index1 = torch.randint(0, 1000, (3000,))
# output1 = scatter_mean(input1, index=index1)
# print("Output shape (Case 1):", output1.shape)

# # Example usage for Case 2 (index skipped, output_dim specified):
# input2 = torch.randn(3000, 144)
# output_dim = [3000]
# output2 = scatter_mean(input2, output_dim=output_dim)
# print("Output shape (Case 2):", output2.shape)

# # Example usage for Case 3 (both spe):
# input = torch.randn(3000, 144)
# index = torch.randint(0, 1000, (3000,))
# output_dim = [3000]

# output = scatter_mean(input, index, output_dim)
# print(output.size())  # Should print torch.Size([1000, 144])