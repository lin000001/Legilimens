import os
import argparse
def args_parser():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--model_path', type=str)
	parser.add_argument('--model_name', type=str, help='Specify the model name')
	parser.add_argument('--user_prompt', type=str, help='Test')
	parser.add_argument('--gpus', type=str)
	args = parser.parse_args()
	return args
args = args_parser()
if args.gpus:
	gpu_list = args.gpus.split(',')
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch.autograd import Variable
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
import random
from fastchat.conversation import get_conv_template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_name, model_path):
	if model_name == 'chatglm3':
		tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
		# import pdb;pdb.set_trace()
		llm = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
		llm = llm.eval()
	elif model_name == 'llama2':
		llm = LlamaForCausalLM.from_pretrained(
							model_path,
							load_in_8bit = False,
							torch_dtype = torch.float16,
							device_map = 'auto'
							# device_map = "cuda:0"
	   )
		tokenizer = LlamaTokenizer.from_pretrained(model_path)
		llm = llm.eval()
	elif model_name == 'vicuna':
		llm = LlamaForCausalLM.from_pretrained(
						model_path,
						load_in_8bit = False,
						torch_dtype = torch.float16,
						device_map = 'auto'
						
					)
		tokenizer = LlamaTokenizer.from_pretrained(model_path)
		llm = llm.eval()
	elif model_name == 'falcon':
		import transformers
		tokenizer = AutoTokenizer.from_pretrained(model_path)
		pipeline = transformers.pipeline("text-generation",
									model=model_path,
									tokenizer=tokenizer,
									torch_dtype=torch.bfloat16,
									# trust_remote_code=True,			
									device_map="auto")
		llm = pipeline
	elif model_name == 'dolly':
		from tools.instruct_pipeline import InstructionTextGenerationPipeline
		tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
		llm_pre = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
		llm_pre = llm_pre.eval()
		generate_text= InstructionTextGenerationPipeline(model=llm_pre, tokenizer=tokenizer)
		llm = generate_text
	else:  
		raise ValueError("Invalid model name.")
	return llm, tokenizer


def query_prompt(llm, tokenizer, user_prompt, model_name):

	if model_name == 'chatglm3':
		response, history, feature, all_layer_feature = llm.chat(tokenizer, user_prompt, history=[])
		print(response)
	elif model_name == 'llama2':
		prompt = '''<s>[INST] <<SYS>>
					You are a helpful assistant. Answer the questions as best as you can.
					<</SYS>>
					{} [/INST]'''
		inputs = tokenizer(prompt.format(user_prompt), return_tensors="pt")
		input_ids = inputs["input_ids"].to(device)
		generate_ids, features, all_layer_feature = llm.generate(input_ids, max_length=8192)
		response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		print(response)
	elif model_name == 'vicuna':
		conv = get_conv_template("vicuna_v1.1")
		conv.append_message(conv.roles[0], user_prompt)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()
		inputs = tokenizer(prompt, return_tensors="pt")
		input_ids = inputs["input_ids"].to(device)
		generate_ids, feature, all_layer_feature = llm.generate(input_ids, max_length=8192)
		response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		print(response)

	elif model_name == 'falcon':
		conv = get_conv_template("falcon")
		conv.append_message(conv.roles[0], user_prompt)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()
		sequences, feature1, all_layer_feature = llm(
			prompt,
			max_length=8192,
			do_sample=True,
			top_k=10,
			num_return_sequences=1,
			eos_token_id=tokenizer.eos_token_id,
		)
		for seq in sequences:
			print(f"Result: {seq['generated_text']}")
	elif model_name == 'dolly':
		conv = get_conv_template("dolly_v2")
		conv.append_message(conv.roles[0], user_prompt)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()
		res, feature1, all_layer_feature  = llm(prompt)
		print(res[0]["generated_text"])



if __name__ == '__main__':
	args = args_parser()

	llm, tokenizer = load_model_and_tokenizer(args.model_name, args.model_path)
	

	query_prompt(llm, tokenizer, args.user_prompt, args.model_name)

	 
	
	
