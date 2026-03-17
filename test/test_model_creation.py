from classifier import *


def test_model_creation():
    model = SimpleCNN()
    print(model)
    print(measure_size(model))
    print(measure_size(SimpleCNN(dtype=torch.float16)))
    print(measure_size(SimpleCNN(h_conv=[], h_fc=[])))



