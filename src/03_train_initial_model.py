from pathlib import Path
import random, numpy as np, pandas as pd, torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RESULTS_DIR=Path('results'); TRAIN_CSV=RESULTS_DIR/'train_metadata.csv'; VAL_CSV=RESULTS_DIR/'val_metadata.csv'
class CropDataset(Dataset):
    def __init__(self,csv_path,label_to_idx):
        self.df=pd.read_csv(csv_path); self.label_to_idx=label_to_idx
        self.transform=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        x=self.transform(Image.open(row['image_path']).convert('RGB'))
        y=self.label_to_idx[row['label']]
        return x, torch.tensor(y,dtype=torch.long)
class SmallCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*16*16,64),nn.ReLU(),nn.Linear(64,num_classes))
    def forward(self,x): return self.net(x)
def evaluate(model,loader):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            preds=model(x).argmax(dim=1); correct+=(preds==y).sum().item(); total+=y.numel()
    return correct/max(total,1)
def main():
    labels=sorted(pd.read_csv(TRAIN_CSV)['label'].unique().tolist()); label_to_idx={l:i for i,l in enumerate(labels)}
    train_loader=DataLoader(CropDataset(TRAIN_CSV,label_to_idx),batch_size=16,shuffle=True)
    val_loader=DataLoader(CropDataset(VAL_CSV,label_to_idx),batch_size=16)
    model=SmallCNN(len(labels)); opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss(); rows=[]
    for epoch in range(1,6):
        model.train(); total_loss=0.0
        for x,y in train_loader:
            opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step(); total_loss+=loss.item()
        avg=total_loss/len(train_loader); acc=evaluate(model,val_loader)
        rows.append({'epoch':epoch,'train_loss':avg,'val_accuracy':acc})
        print(f'Epoch {epoch}: loss={avg:.4f}, val_accuracy={acc:.3f}')
    metrics=pd.DataFrame(rows); metrics.to_csv(RESULTS_DIR/'training_metrics.csv',index=False)
    plt.figure(); plt.plot(metrics['epoch'],metrics['train_loss'],marker='o'); plt.xlabel('Epoch'); plt.ylabel('Training Loss'); plt.title('Milestone 1 Initial Model Loss'); plt.savefig(RESULTS_DIR/'training_loss.png',bbox_inches='tight')
    torch.save({'model_state_dict':model.state_dict(),'label_to_idx':label_to_idx},RESULTS_DIR/'initial_model.pt')
if __name__=='__main__': main()
