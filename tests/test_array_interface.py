import pytest
import torch

from torchstream.sequence.sequential_array import get_shape_and_array_interface


def test_get_shape_and_array_interface():
    ## First and second overloads
    shape, arr_if = get_shape_and_array_interface(10, -1, dtype=torch.int32, device="cuda:0")
    assert shape == (10, -1)
    assert arr_if.dtype == torch.int32
    assert arr_if.device == torch.device("cuda:0")

    shape, arr_if = get_shape_and_array_interface((10, -1), torch.int32, device="cuda:0")
    assert shape == (10, -1)
    assert arr_if.dtype == torch.int32
    assert arr_if.device == torch.device("cuda:0")

    shape, arr_if = get_shape_and_array_interface((10, -1), "cuda:0")
    assert shape == (10, -1)
    assert arr_if.dtype == torch.float32
    assert arr_if.device == torch.device("cuda:0")

    ## Dimensions
    shape, _ = get_shape_and_array_interface(10, -1, 30)
    assert shape == (10, -1, 30)

    shape, _ = get_shape_and_array_interface(10, -2, 30)
    assert shape == (10, -2, 30)

    shape, _ = get_shape_and_array_interface(10, 20, 30)
    assert shape == (10, 20, 30)

    with pytest.raises(ValueError):
        get_shape_and_array_interface(10, -1, -1)

    with pytest.raises(ValueError):
        get_shape_and_array_interface(10, -1, 0)

    with pytest.raises(ValueError):
        get_shape_and_array_interface(0, 5)

    ## Third overload
    shape, arr_if = get_shape_and_array_interface(torch.randn(10, 20, 30), seq_dim=1)
    assert shape == (10, -1, 30)
    assert arr_if.dtype == torch.float32
    assert arr_if.device == torch.device("cpu")

    shape, arr_if = get_shape_and_array_interface(torch.randn(10, 20, 30), 1)
    assert shape == (10, -1, 30)
    assert arr_if.dtype == torch.float32
    assert arr_if.device == torch.device("cpu")
