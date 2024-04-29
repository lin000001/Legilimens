import os  
import torch  
from torch.utils.data import Dataset  
from tqdm import tqdm  
import pandas as pd
 
class PromptBinaryLazyDataset(Dataset):    
    def __init__(self, data, label):    
        self.data = data    
        self.label = label    
    
    @classmethod    
    def lazy_load(cls, data_path):    
        data = []    
        target = []    
        print(f"Loading data lazily...")    
        for file in tqdm(os.listdir(data_path)):    
            content = torch.load(os.path.join(data_path, file))    
            data.extend(content['feature'])    
            target.extend(content['target'])    
            yield data, target   
    
    def __len__(self):    
        return len(self.data)    
    
    def __getitem__(self, idx):    
        feature = torch.Tensor(self.data[idx][1][-1])  # final layer    
        label = torch.tensor(1 if self.label[idx] == False else 0, dtype=torch.float32)      
        feature = feature.squeeze(0)    
        label = label.squeeze(0)    
        return feature, label      
    

class multi_layercommonDataset(Dataset):  
    def __init__(self, data, label, layer):
        self.data = data
        self.label = label
        self.layer = layer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): 
        label = self.label[idx] 
        # feature = feature.squeeze(0)  
        label = label.squeeze(0)   
        # import pdb;pdb.set_trace()
        features = []
        for layer_data in self.data[idx][1][-self.layer:]:  
            feature = torch.Tensor(layer_data)  
            feature = feature.squeeze(0)  
            features.append(feature) 
        feature_multi = torch.cat(features, dim= 1)
        return feature_multi, label 
    

from sklearn.utils import resample  
  

class multi_layerpromptDataset(Dataset):  
    def __init__(self, data, label, layer):
        self.data = data
        self.label = label
        self.layer = layer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): 
        label = torch.Tensor([0 if self.label[idx] == torch.tensor([True]) else 1]) 
        # feature = feature.squeeze(0)  
        label = label.squeeze(0)   
        features = []
        for layer_data in self.data[idx][1][-self.layer:]:  
            feature = torch.Tensor(layer_data)  
            feature = feature.squeeze(0)  
            features.append(feature) 
        feature_multi = torch.cat(features, dim= 1)
        return feature_multi, label


        
# The following is the 'extra_feature' part, and the 'train' and 'test' do not use the Classes below.
# extra_feature.py CustomDataset-->jigsaw  PromptDataset-->beaver,oig      
class CustomDataset(Dataset):
    def __init__(self, csv_file, features_column, labels_column):
        self.df = pd.read_csv(csv_file)
        self.features = self.df[features_column].values
        self.labels = self.df[labels_column].values
        self.string_label = ['innocent', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'labels': self.tovalue(self.labels[idx])}
        
        return sample
    
    def tovalue(self, label):
        # import pdb;pdb.set_trace()
        if label == 'innocent':
            return 0
        else:
            return 1
class OigfeatureDataset(Dataset):
    def __init__(self, csv_file, features_column, labels_column):
        self.df = pd.read_csv(csv_file)
        self.features = self.df[features_column].values
        self.labels = self.df[labels_column].values
        self.string_label = ['casual', 'possibly needs caution', 'needs caution', 'needs intervention']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'labels': self.tovalue(self.labels[idx])}
        
        return sample
    
    def tovalue(self, label):
        # import pdb;pdb.set_trace()
        if label == 'casual' or label == 'possibly needs caution':
            return 0
        else:
            return 1


class PromptDataset(Dataset):
    def __init__(self, csv_file, features_column, labels_column):
        self.df = pd.read_csv(csv_file)
        self.features = self.df[features_column].values
        self.labels = self.df[labels_column].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'labels': self.labels[idx] }
        return sample
class JailbreakDataset(Dataset):
    def __init__(self, csv_file, features_column,response_column, labels_column):
        self.df = pd.read_csv(csv_file)
        self.features = self.df[features_column].values
        self.response = self.df[response_column].values

        self.labels = self.df[labels_column].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if pd.isna(self.response[idx]):
            self.response[idx] = 'None'
        sample = {'features': self.features[idx],'response': self.response[idx], 'labels': self.labels[idx] }
        return sample
    
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]][0]
        label = self.dataset[self.indices[idx]][1]
        index = self.indices[idx]
        return data, label, index
    
    def __len__(self):
        return len(self.indices)
    
