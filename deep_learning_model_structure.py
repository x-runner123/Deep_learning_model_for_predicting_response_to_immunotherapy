import numpy as np
import random
import torch
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing  import MinMaxScaler

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
os.environ['PYTHONHASHSEED']=str(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

#selected genes derived from four levels 
layer0_names=pd.read_csv("MCC_top30_gene_names.csv")
layer1_names=pd.read_csv("PDL1_gene_set.csv")
layer2_names=pd.read_csv("genes_for_predicting_TMB_and_MMR.csv")
layer3_names=pd.read_csv("core_genes_for_RFE.csv")
selected_genes=set(list(layer0_names["x"]))|set(list(layer1_names["x"]))|set(list(layer2_names["x"]))|set(list(layer3_names["x"]))
selected_genes1=list(layer0_names["x"])+list(layer1_names["x"])+list(layer2_names["x"])+list(layer3_names["x"])
selected_genes2=np.array(selected_genes1)
selected_genes3=selected_genes2.reshape(12,8)

#read training data
gene_expression_data=pd.read_hdf("Pan-cancer_TPM.h5")
phenotype_data=pd.read_csv("Pan-cancer_phenotype_data.csv",index_col=0)

gene_expression_data1=gene_expression_data.loc[selected_genes1,].T
gene_expression_data2=np.array(gene_expression_data1)
gene_expression_data3=gene_expression_data2
gene_expression_data4=torch.tensor(gene_expression_data3)

gene_expression_data5 = np.log2(gene_expression_data4+1)
all_96_gene_mean_expression = torch.sum(gene_expression_data5.reshape(-1))/8710

class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labelsA, labelsB):
        self.inputs = inputs
        self.labelsA = labelsA
        self.labelsB = labelsB
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labelsA = self.labelsA[index]
        labelsB = self.labelsB[index]
        return inputs, labelsA, labelsB

minmaxscaler = MinMaxScaler()
pan_cancer_TMB=np.log10(np.array(phenotype_data['TMB'])+0.001).reshape(-1,1)
pan_cancer_TMB1=minmaxscaler.fit_transform(pan_cancer_TMB)

minmaxscaler = MinMaxScaler()
pan_cancer_PD_L1=np.log2(np.array(phenotype_data['PD_L1'])+1).reshape(-1,1)
pan_cancer_PD_L1_1=minmaxscaler.fit_transform(pan_cancer_PD_L1)

trainset1=MultitaskDataset(gene_expression_data4,torch.tensor(pan_cancer_TMB1),torch.tensor(pan_cancer_PD_L1_1))

import torch.nn as nn
import torch.nn.functional as F

class GeneTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout):
        super(GeneTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, dropout) 
            for _ in range(num_layers)])
        
        #Average
        self.avg=nn.AvgPool2d(kernel_size=(1,8))
        
        # Output layer
        self.output_layer0 = nn.Linear(12, 4)
        self.output_layer1 = nn.Linear(4, 1)
        self.output_layer2 = nn.Linear(12, 4)
        self.output_layer3 = nn.Linear(4, 1)

    def forward(self, x):
        # Embedding layer
        x=x.view(-1,12,self.hidden_size)
        x1=x
        x_copy=x
        x1=torch.log2(x1+1)
        gene_ratio=torch.sum(x_copy.reshape(-1))/all_96_gene_mean_expression
        x1=x1/gene_ratio
        x1=nn.LayerNorm(self.hidden_size)(x1)

        # Transformer layers
        for layer in self.transformer_layers:
            x1= layer(x1)

        x1 = self.avg(x1)
        x1 =x1.view(x.size(0),-1)
        hidden_x1 = x1
        x1_A = self.output_layer0(x1)
        x1_A = nn.ReLU()(x1_A)
        x1_A = self.output_layer1(x1_A)
        x1_A = 4*nn.Tanh()(x1_A)
        x1_A = torch.sigmoid(x1_A)
        x1_B = self.output_layer2(x1)
        x1_B = nn.ReLU()(x1_B)
        x1_B = self.output_layer3(x1_B)
        x1_B = 4*nn.Tanh()(x1_B)
        x1_B = torch.sigmoid(x1_B)
        final_outputs = 0.5*x1_A+0.5*x1_B

        return x1_A, x1_B, final_outputs

    
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Multi-head attention layer
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout,batch_first=True)

        # Feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size))

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        


    def forward(self, x):
        # Multi-head attention layer
        residual = x
        x, _ = self.multi_head_attention(x, x, x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm1(x + residual)

        # Feedforward layer
        residual = x
        x = self.feedforward(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_norm2(x + residual)

        return x
    


torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
os.environ['PYTHONHASHSEED']=str(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

hidden_size = 8
num_layers = 2
num_heads = 2
dropout = 0.2
k = 3
epochs = 500
batch_size = 128
lr=0.001

kf = KFold(n_splits=k, shuffle=True, random_state=123)

gene_transformer = GeneTransformer(hidden_size, num_layers, num_heads, dropout)
criterion_A = torch.nn.MSELoss(size_average=None, reduce=None, reduction="sum")
criterion_B = torch.nn.MSELoss(size_average=None, reduce=None, reduction="sum")
optimizer = torch.optim.Adam(gene_transformer.parameters(), lr=0.001)

loss_observer=[[],[],[]]

for fold, (train_indices, val_indices) in enumerate(kf.split(trainset1)):
    
    # split the data
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(trainset1, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(trainset1, batch_size=batch_size, sampler=val_sampler)

    # initialization
    net = gene_transformer
    
    #train
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labelsA ,labelsB = data
            inputs = inputs.to(torch.float32)
            optimizer.zero_grad()
            outputsA_0 = net(inputs)[0]
            outputsA_1 = outputsA_0.view(-1,1)
            labelsA = labelsA.view(-1,1)
            outputsB_0 = net(inputs)[1]
            outputsB_1 = outputsB_0.view(-1,1)
            labelsB = labelsB.view(-1,1)
            lossA = criterion_A(outputsA_1,labelsA.float())
            lossB = criterion_B(outputsB_1,labelsB.float())
            loss = 0.5*lossA+0.5*lossB
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Fold [%d]/[%d] Epoch [%d]/[%d] Loss: %.3f" % (fold+1, k, epoch+1, epochs, running_loss/(i+1)))
        loss_observer[fold].append(running_loss/(i+1))
    
    correct = 0
    total = 0    
    net.eval()
    with torch.no_grad():
        for data in val_loader:
            inputs, labelsA, labelsB = data
            inputs = inputs.to(torch.float32)
            outputs0 = net(inputs)[2]
    print(outputs0)
    