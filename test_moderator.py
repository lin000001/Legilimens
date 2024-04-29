import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
# from dataset import CustomDataset
from torch.utils.data import DataLoader
import os
from torch.utils.data import TensorDataset
import random
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tools.classifier_architecture import *
from tools.dataset import multi_layerpromptDataset,multi_layercommonDataset
import wandb
import numpy as np
from tools.classifier_architecture import *

def args_parser():  
	parser = argparse.ArgumentParser(description='')  
	parser.add_argument('--modellayer', type=int, default=3, help='Number of layers for the neural network')
	parser.add_argument('--featurelayer', type=int, default=1)
	parser.add_argument('--modelname', type=str)  
	parser.add_argument('--result_path', type=str)  
	parser.add_argument('--test_dataset', type=str, help='test target dataset')  

	parser.add_argument('--gpus', type=str)
	
	args = parser.parse_args()  
	os.makedirs(args.result_path, exist_ok=True)
	
	return args
args = args_parser()
if args.gpus:
	gpu_list = args.gpus.split(',')
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
	

def load_dataset(dataset_name, dataset_path, featurelayer):

	dataset_name = dataset_name.lower()
	if dataset_name == "beaver":  
		beaver_path = dataset_path + '/beaver/test'
		beaverdata , beaver_target = cat_feature(beaver_path)
		testset = multi_layerpromptDataset(beaverdata , beaver_target, featurelayer)
	elif dataset_name == "oig":  
		oig_path = dataset_path + '/oig/test'
		oigdata , oig_target = cat_feature(oig_path)
		testset = multi_layercommonDataset(oigdata , oig_target, featurelayer) 
	elif dataset_name == "jigsaw":  
		jigsaw_path = dataset_path + '/jigsaw/test'
		jigsawdata , jigsaw_target = cat_feature(jigsaw_path)
		testset = multi_layercommonDataset(jigsawdata , jigsaw_target, featurelayer)
	elif dataset_name == "BEA_adv":
		jailbreak_path = dataset_path + '/BEA_adv/test'
		jailbreakdata, jailbreak_target = cat_feature(jailbreak_path)
		testset = multi_layerpromptDataset(jailbreakdata, jailbreak_target, featurelayer)
	elif dataset_name == "BAG":
		jailbreak_path = dataset_path + '/BAG/test'
		jailbreakdata, jailbreak_target = cat_feature(jailbreak_path)
		testset = multi_layercommonDataset(jailbreakdata, jailbreak_target, featurelayer)
	else:  
		raise NotImplementedError 
	return testset	

def test(model, test_loader, result_path, modellayer, featurelayer, model_name, jailbreak_type):
	model.eval()
	test_loss = 0
	correct = 0
	total = 0
	correct_binary = 0
	y_true = []  
	y_pred = []  
	y_probs = []
	criterion = nn.CrossEntropyLoss()
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.cuda(), target.cuda()
			output = torch.squeeze(model(data.to(torch.float32)),1)
			target = target.to(torch.int64)
			test_loss += F.cross_entropy(output, target, size_average=False).item() 
			total += data.shape[0]
			pred = torch.max(output, 1)[1]
			probs = F.softmax(output, dim=1)
			y_scores = probs[:, 1]
			correct += pred.eq(target.view_as(pred)).sum().item()
			y_true.extend(target.cpu().numpy())  
			y_pred.extend(pred.cpu().numpy())
			y_probs.extend(y_scores.cpu().numpy())
	
	test_loss = test_loss / total
	acc = 100. * correct / total
	acc_binary = 100. * correct_binary / total    
	plot_roc_auc(y_true, y_probs, result_path, modellayer, featurelayer, model_name, jailbreak_type)
	cm = confusion_matrix(y_true, y_pred)  
	plt.figure(figsize=(10, 8))  
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=2, yticklabels=2)  
	plt.xlabel('Predicted label')  
	plt.ylabel('True label') 
	save_path= os.path.join(result_path, f"cm_{model_name}_{jailbreak_type}_{modellayer}_{featurelayer}_{acc}.png")
	plt.savefig(save_path) 
	print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}, Accuracy_binary: {:.4f}\n'
		  .format(test_loss, acc, acc_binary))
	return acc, test_loss

from sklearn.metrics import roc_curve, auc  
  
def plot_roc_auc(true_labels, predictions, result_path, modellayer, featurelayer, model_name, jailbreak_type):  
	
	fpr, tpr, thresholds = roc_curve(true_labels, predictions, drop_intermediate=False)    
	roc_auc = auc(fpr, tpr)    
	y = tpr - fpr
	Youden_index = np.argmax(y)  
	optimal_threshold = thresholds[Youden_index]
	point = [fpr[Youden_index], tpr[Youden_index]]
	plt.figure()  
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)  
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
	plt.xlim([0.0, 1.0])  
	plt.ylim([0.0, 1.05]) 
	plt.axvline(x=point[0], color='r', linestyle='--')
	plt.axhline(y=point[1], color='r', linestyle='--') 
	plt.text(point[0], point[1], f'Threshold:{optimal_threshold:.4f}')
	plt.xlabel('False Positive Rate')  
	plt.ylabel('True Positive Rate')  
	plt.title('Receiver Operating Characteristic (ROC) Curve')  
	plt.legend(loc="lower right")  
	save_path = os.path.join(result_path, f"auc_{model_name}_{jailbreak_type}_{modellayer}_{featurelayer}_{roc_auc}.png")
	plt.savefig(save_path)
	print("auc:",roc_auc)
	
def cat_feature(data_path):
	data = []
	target = []
	print(f"Loading data...")
	for file in tqdm(os.listdir(data_path)):
		content = torch.load(os.path.join(data_path,file))
		data.extend(content['feature'])
		target.extend(content['target'])
	return data, target




if __name__ == '__main__':

	args = args_parser()
	print("Test dataset:", args.test_dataset)
	os.makedirs(args.result_path, exist_ok=True)
	if args.modelname == 'falcon':
		model = create_model(layer=args.modellayer, input_dim=4544*args.featurelayer)
	else:
		model = create_model(layer=args.modellayer, input_dim=4096*args.featurelayer)

	if args.modelname == 'llama2':
		model.load_state_dict(torch.load('./'))
	if args.modelname == 'vicuna':
		model.load_state_dict(torch.load('./'))
	if args.modelname == 'chatglm3':
		model.load_state_dict(torch.load('./'))
	if args.modelname == 'dolly':
		model.load_state_dict(torch.load('./'))
	if args.modelname == 'falcon':
		model.load_state_dict(torch.load('./'))

	dataset_path = ''
	testset = load_dataset(args.test_dataset, f'dataset_path/{args.modelname}', args.featurelayer)

	model.cuda()  
	model.eval()
	testloader = DataLoader(testset, batch_size=2000, shuffle=False)
	test(model, testloader, args.result_path, args.modellayer, args.featurelayer,args.modelname, args.test_dataset)
