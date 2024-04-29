import torch.nn.functional as F
import torch
import torch.nn as nn
import argparse  

class OneLayerClassifier(nn.Module):  
    def __init__(self, dim):  
        super(OneLayerClassifier, self).__init__()  
        self.fc1 = nn.Linear(dim, 2)  
  
    def forward(self, x):  
        x = self.fc1(x)  
        return x  

class ThreeLayerClassifier(nn.Module):  
    def __init__(self, dim):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(dim , 1024)
        self.fc2 = nn.Linear(1024 , 512)
        self.fc3 = nn.Linear(512 , 2)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x         

class FiveLayerClassifier(nn.Module):  
    def __init__(self, dim):  
        super(FiveLayerClassifier, self).__init__()  
        self.fc1 = nn.Linear(dim, 1024)  
        self.fc2 = nn.Linear(1024, 512)  
        self.fc3 = nn.Linear(512, 256)  
        self.fc4 = nn.Linear(256, 128)  
        self.fc5 = nn.Linear(128, 2)  
        self.dropout = nn.Dropout(0.7)  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc3(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc4(x))  
        x = self.dropout(x)  
        x = self.fc5(x)  
        return x  
  
class SevenLayerClassifier(nn.Module):  
    def __init__(self, dim):  
        super(SevenLayerClassifier, self).__init__()  
        self.fc1 = nn.Linear(dim, 2048)  
        self.fc2 = nn.Linear(2048, 1024)  
        self.fc3 = nn.Linear(1024, 512)  
        self.fc4 = nn.Linear(512, 256)  
        self.fc5 = nn.Linear(256, 128)  
        self.fc6 = nn.Linear(128, 64)  
        self.fc7 = nn.Linear(64, 2)  
        self.dropout = nn.Dropout(0.7)  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc3(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc4(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc5(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc6(x))  
        x = self.dropout(x)  
        x = self.fc7(x)  
        return x  
  
class NineLayerClassifier(nn.Module):  
    def __init__(self, dim):  
        super(NineLayerClassifier, self).__init__()  
        self.fc1 = nn.Linear(dim, 4096)  
        self.fc2 = nn.Linear(4096, 2048)  
        self.fc3 = nn.Linear(2048, 1024)  
        self.fc4 = nn.Linear(1024, 512)  
        self.fc5 = nn.Linear(512, 256)  
        self.fc6 = nn.Linear(256, 128)  
        self.fc7 = nn.Linear(128, 64)  
        self.fc8 = nn.Linear(64, 32)  
        self.fc9 = nn.Linear(32, 2)  
        self.dropout = nn.Dropout(0.7)  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc3(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc4(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc5(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc6(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc7(x))  
        x = self.dropout(x)  
        x = F.relu(self.fc8(x))  
        x = self.dropout(x)  
        x = self.fc9(x)  
        return x 


def create_model(layer, input_dim):  
    if layer == 1:  
        model = OneLayerClassifier(dim=input_dim)  
    elif layer == 3:  
        model = ThreeLayerClassifier(dim=input_dim)  
    elif layer == 5:  
        model = FiveLayerClassifier(dim=input_dim)  
    elif layer == 7:  
        model = SevenLayerClassifier(dim=input_dim)  
    elif layer == 9:  
        model = NineLayerClassifier(dim=input_dim)  
    else:  
        raise ValueError("Invalid layer value. Please choose from 1, 3, 5, 7, or 9.")  
  
    return model  
  

  
# if __name__ == "__main__":  
#     main() 