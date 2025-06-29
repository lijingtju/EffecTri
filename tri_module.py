import os
import pandas as pd
import torch
from mlp_net import MLP

class TriModule():
    def __init__(self, params:str = None, lr:float = 0.005, m:float = 0.9, w:float = 5e-4, input_dim:int = 739, hidden_dim:int = 128, num_classes:int = 3, p:float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, p=p).to(device=self.device)
        self.params = params
        self.model = self.loadParams()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=m, weight_decay=w)
        self.criterion = torch.nn.CrossEntropyLoss()

    def loadParams(self):
        if self.params is None:
            return self.model
        checkpoint = torch.load(self.params, weights_only=False)
        ckp_keys = list(checkpoint)
        model_sd = self.model.state_dict()
        for ckp_key in ckp_keys:
            model_sd[ckp_key] = checkpoint[ckp_key]
        self.model.load_state_dict(model_sd,)
        return self.model
    

if __name__ == '__main__':
    tri = TriModule("./models/model_epoch_156.pt")
