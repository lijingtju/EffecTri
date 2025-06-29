from tri_module import TriModule
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import classification_report
from tri_utils import ensure_directory
import time

class TriTrain():
    def __init__(self, train_data_path:str, val_data_path:str, test_data_path:str, resume:str = None, batch_size:int = 16):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.y_train = self.train_data[1].values.flatten()
        self.x_train = self.train_data.iloc[:, 2:].values
        
        self.val_data =  pd.read_csv(val_data_path, header=None)
        self.y_val = self.val_data[1].values.flatten()
        self.X_val = self.val_data.iloc[:, 2:].values

        self.test_data = pd.read_csv(test_data_path, header=None)
        self.y_test = self.test_data[1].values.flatten()
        self.X_test = self.test_data.iloc[:, 2:].values

        self.input_dim = self.train_data.shape[1] - 2

        self.train_dataset = TensorDataset(torch.tensor(self.x_train, dtype=torch.float32),
                              torch.tensor(self.y_train, dtype=torch.long))
        self.val_dataset = TensorDataset(torch.tensor(self.X_val, dtype=torch.float32),
                                    torch.tensor(self.y_val, dtype=torch.long))
        self.test_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32),
                                    torch.tensor(self.y_test, dtype=torch.long))

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size)


        self.triModule = TriModule(params=resume, input_dim=self.input_dim)


    def train(self, num_epochs:int = 500):
        for epoch in range(1, num_epochs + 1):
            self.triModule.model.train()
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.triModule.device), yb.to(self.triModule.device)
                self.triModule.optimizer.zero_grad()
                pred = self.triModule.model(xb)
                loss = self.triModule.criterion(pred, yb)
                loss.backward()
                self.triModule.optimizer.step()
            yield epoch

    def val(self, metrics_dir:str, epoch:int):
        ensure_directory(metrics_dir)
        self.triModule.model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.triModule.device)
                output = self.triModule.model(xb)
                val_preds.extend(output.argmax(dim=1).cpu().numpy())
                val_true.extend(yb.numpy())
        report = classification_report(val_true, val_preds, target_names=["0", "1", "2"], digits=4, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(f"{metrics_dir}/val_epoch_{epoch}.csv")

    def test(self, metrics_dir:str):
        ensure_directory(metrics_dir)
        self.triModule.model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb = xb.to(self.triModule.device)
                out = self.triModule.model(xb)
                test_preds.extend(out.argmax(dim=1).cpu().numpy())
                test_true.extend(yb.numpy())

        report = classification_report(test_true, test_preds, target_names=["0", "1", "2"], digits=4, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(f"{metrics_dir}/final_test_report_{int(time.time()*1000)}.csv")

    def save(self, saveDir:str, epoch:int):
        ensure_directory(saveDir)
        torch.save(self.triModule.model.state_dict(), f"{saveDir}/model_epoch_{epoch}.pt")

if __name__ == '__main__':
    triTrain = TriTrain(train_data_path="./datasets/ESM_AAC_DPC_train_scaled.csv",
                        val_data_path="./datasets/ESM_AAC_DPC_val_scaled.csv",
                        test_data_path="./datasets/ESM_AAC_DPC_test_scaled.csv")
    print("start train...\n")
    for epoch in triTrain.train():
        print("[ train ] {}".format(epoch))
        triTrain.save("./weights", epoch=epoch)
        triTrain.val(metrics_dir="./log/val", epoch=epoch)

    print("start test...\n")
    triTrain.test(metrics_dir="./log/test")
    print(" = = = Finish train = = = ")