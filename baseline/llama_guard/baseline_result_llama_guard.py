import json
import pandas as pd
from tqdm import tqdm
import os
correct = 0
total = 0
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

"""
python baseline_result.py --baseline_name openai --dataset_name beaver --metric acc --gpus 0
"""

def plot_roc_auc(true_labels, predictions,baseline_name, dataset_name, result_path):  
	
	fpr, tpr, thresholds = roc_curve(true_labels, predictions, drop_intermediate=False)    
	roc_auc = auc(fpr, tpr)    
	y = tpr - fpr
	Youden_index = np.argmax(y)  # Only the first occurrence is returned.
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
	save_path = f"{result_path}/auc_{baseline_name}_{dataset_name}.png" 
	plt.savefig(save_path) 

# path = 'XXX/llama_guard/predictions_llama_guard_bag_adv.json'
# path = 'XXX/llama_guard/predictions_llama_guard_bag.json'
# path = 'XXX/llama_guard/predictions_llama_guard_beaver_adv.json'
# path = 'XXX/llama_guard/predictions_llama_guard_beaver_all.json'
# path = 'XXX/llama_guard/predictions_llama_guard_hatexpslain.json'
path = 'XXX/llama_guard/predictions_llama_guard_jigsaw.json'
# path = 'XXX/llama_guard/predictions_llama_guard_mhs.json'
# path = 'XXX/llama_guard/predictions_llama_guard_oig.json'
	
with open(path, 'r') as file:  
	data = json.load(file)  
	gt = data[0]['gt']
	y_pred = data[2]['y_pre'] 
	# gt = [0 if bool(i) else 1 for i in gt]
	print(len(gt))
	tn, fp, fn, tp = confusion_matrix(gt, y_pred).ravel()  
	fpr = fp / (fp + tn)  
	fnr = fn / (fn + tp)
	accuracy = accuracy_score(gt, y_pred)  
	print(f'Accuracy: {accuracy}')   
	auc_score = roc_auc_score(gt, y_pred)  
	print(f'AUC: {auc_score}')
	print(f'FPR: {fpr}')
	print(f'FNR: {fnr}')




	
