import torch
from tri_module import TriModule
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from tri_utils import ensure_directory
import time
import os

class TriEvaluate():
    def __init__(self, data_path:str, resume:str, batch_size:int = 16):
        self.data = pd.read_csv(data_path, header=None)
        self.x = self.data.iloc[:, 1:].values

        self.input_dim = self.data.shape[1] - 1
        self.dataset = TensorDataset(torch.tensor(self.x, dtype=torch.float32))
        self.evaluate_loader = DataLoader(self.dataset, batch_size=batch_size)
        self.triModule = TriModule(params=resume, input_dim=self.input_dim)

    def evaluate(self, metrics_dir:str):
        ensure_directory(metrics_dir)
        self.triModule.model.eval()
        test_preds = []
        with torch.no_grad():
            for xb in self.evaluate_loader:
                xb = xb[0].to(self.triModule.device)
                out = self.triModule.model(xb)
                test_preds.extend(out.argmax(dim=1).cpu().numpy())
        
        result = "result"
        for ret in test_preds:
            result = "{}\n{}".format(result, int(ret))
        
        with open(os.path.join(metrics_dir, "evaluate_report_{}.csv".format(int(time.time()*1000))), "w", encoding='utf-8') as f:
            f.write(result)




if __name__ == '__main__':
    triEvaluate = TriEvaluate(data_path="./datasets/T346SE_test.csv", resume="./models/model_epoch_156.pt")
    triEvaluate.evaluate("./log/evaluate")