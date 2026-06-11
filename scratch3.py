import torch
import torch.nn as nn

class SharedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = SharedModule()
        self.analyzer = Analyzer(self.shared)

class Analyzer(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.other = nn.Linear(5, 5)

m = MyModel()
for n, p in m.named_parameters():
    print(n)
