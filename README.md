# Legilimens

## Introduction
This is the official code for "[Legilimens: Practical and Unified Content Moderation for Large Language Model Services](https://arxiv.org/pdf/2408.15488)" (Accepted by ACM CCS 2024).

## Preperation

You can build the required environment by running:

```bash
pip install -r requirements.txt
```


Put dataset in ``./datasets``.

Put pre-trained moderator in ``./models_IO_moderation``.
## Usage
The whole project contains three tasks: 
+ I-moderation.
+ O-moderation.
+ IO-moderation.

Here are five host LLMs.
### LLaMA2
Here is a guide on how to deploy our moderator to LLaMA2：
+ First, modify the PyTorch-transformers as shown in our code ``./modeling_llm/LLaMA2/modeling_llama2.py`` and ``./modeling_llm/utils.py``.
+ Second, load the moderator using the path ``./models_IO_moderation/llama2``.
+ Finally, you can use `python test_main.py --model_name llama2 --model_path ./llama2 --user_prompt "Hi" --gpus 0` to moderate the content.

The example of running LLaMA2 equipped with our moderator.

```python
from transformers import LlamaForCausalLM, LlamaTokenizer``
from fastchat.conversation import get_conv_template
model = LlamaForCausalLM.from_pretrained(
                        "./Llama-2-7b-chat-hf",
                        load_in_8bit = False,
                        torch_dtype = torch.float16,
                        device_map = "auto"
                    )
tokenizer = LlamaTokenizer.from_pretrained("./Llama-2-7b-chat-hf")
model = model.eval()
user_prompt = ''
conv = get_conv_template("llama-2")
conv.append_message(conv.roles[0], user_prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
# Generate
# The all_layer_feature is the feature that we moderate.
with torch.no_grad():
    generate_ids, _ , all_layer_feature = model.generate(input_ids, max_length=8192)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

### Vicuna

Use the same steps as llama2.

Here is a guide on how to deploy our moderator to Vicuna：
+ First, modify the PyTorch-transformers as shown in our code ``./modeling_llm/LLaMA2/modeling_llama2.py`` and ``./modeling_llm/utils.py``.
+ Second, load the moderator using the path ``./models_IO_moderation/vicuna``.
+ Finally, you can use `python test_main.py --model_name vicuna --model_path ./vicuna --user_prompt "Hi" --gpus 0` to moderate the content.

### ChatGLM3

Here is a guide on how to deploy our moderator to ChatGLM3：
+ First, modify the PyTorch-transformers as shown in our code ``./modeling_llm/LLaMA2/modeling_chatglm.py`` and ``./modeling_llm/utils.py``.
+ Second, load the moderator using the path ``./models_IO_moderation/chatglm3``.
+ Finally, you can use `python test_main.py --model_name chatglm3 --model_path ./chatglm3 --user_prompt "Hi" --gpus 0` to moderate the content.

The example of running LLaMA2 equipped with our moderator.

```python
from transformers import AutoTokenizer, AutoModel
import os
import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tokenizer = AutoTokenizer.from_pretrained("./chatglm3", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm3", trust_remote_code=True, device='cuda')
model = model.eval()
user_prompt = ''
response, history, features, all_features = model.chat(tokenizer, prompt, history=[])
print(response)
```

### Falcon

Here is a guide on how to deploy our moderator to Falcon：
+ First, modify the PyTorch-transformers as shown in our code ``./modeling_llm/falcon/modeling_falcon.py`` and ``./modeling_llm/utils.py``.
+ Second, load the moderator using the path ``./models_IO_moderation/falcon``.
+ Finally, you can use `python test_main.py --model_name falcon --model_path ./falcon --user_prompt "Hi" --gpus 0` to moderate the content.


### Dolly

Here is a guide on how to deploy our moderator to Dolly：
+ First, modify the PyTorch-transformers as shown in our code ``./modeling_llm/dolly/modeling_gpt_neox.py`` and ``./modeling_llm/utils.py``.
+ Second, load the moderator using the path ``./models_IO_moderation/dolly``.
+ Finally, you can use `python test_main.py --model_name dolly --model_path ./dolly --user_prompt "Hi" --gpus 0` to moderate the content.



