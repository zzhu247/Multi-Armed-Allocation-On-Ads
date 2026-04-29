import torch
# from src.pipeline import Normalizer

class Normalizer:
    def __init__(self):
        pass
    
    def normalize(self, input_tensor):
        if input_tensor.dim() == 0 or input_tensor.numel() == 0:
            return input_tensor
        else:
            sum_tensor = torch.sum(input_tensor, dim= input_tensor.dim() - 1, keepdim=True)

            if sum_tensor.eq(0).any():
                return input_tensor
            return input_tensor / sum_tensor

def test_normal_input():
    input_1 = torch.tensor([[3,2,1], [1.0, 2.0, 3.0]])
    print(input_1.dim())
    print(input_1)
    normalizer_test = Normalizer()
    output_1 = normalizer_test.normalize(input_1)
    assert torch.allclose(output_1, torch.tensor([[0.5, 0.3333, 0.1667], [0.1667, 0.3333, 0.5]]), atol=1e-4)

def test_empty_input():
    input_2 = torch.tensor([])
    print(input_2.dim())
    normalizer_test = Normalizer()
    output_2 = normalizer_test.normalize(input_2)
    assert torch.allclose(output_2, torch.tensor([]), atol=1e-4)

def test_zero_sum_input():
    input_3 = torch.tensor(0)
    print(input_3.dim())
    normalizer_test = Normalizer()
    output_3 = normalizer_test.normalize(input_3)
    assert torch.allclose(output_3, torch.tensor(0), atol=1e-4)
