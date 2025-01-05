import json
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import threading
import math
import pandas as pd  
from googleapiclient import discovery

API_KEY = 'xxx'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def completion_with_backoff(**kwargs):
#     # return openai.Completion.create(**kwargs)
#     return openai.ChatCompletion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def moderation_with_backoff(input):
    return client.comments().analyze(body=input).execute()

def create_thread(m, gt, result_list, lock):
    # result = gpt_35_api_stream([m])
    result = {"input": None,
              "groundtruth": None,
              "perspective": None}
    analyze_request = {
  'comment': { 'text': m },
  'requestedAttributes': {'TOXICITY': {}}
}
    response = moderation_with_backoff(
        input=analyze_request
    )
    outcome = json.dumps(response, indent=2)
    result['perspective'] = outcome
    # import pdb; pdb.set_trace()
    result['input'] = m
    result['groundtruth'] = gt
    with lock:
        result_list.append(result)

def prespect_api_batch_thread(messages: list, label: list, save_path: str):
    # answers = []
    MAX_NUM_THREAD = 50
    for idx in tqdm(range(math.ceil(len(messages)//MAX_NUM_THREAD))):
        this_batch = messages[idx*MAX_NUM_THREAD: (idx+1)*MAX_NUM_THREAD]
        this_batch_gt = label[idx*MAX_NUM_THREAD: (idx+1)*MAX_NUM_THREAD]

        this_answer = []
        lock = threading.Lock()
        threads = []
        
        for m, gt in zip(this_batch, this_batch_gt):
            t = threading.Thread(target=create_thread, args=(m, gt, this_answer, lock))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        answers = this_answer
        # print("answer:", answers)
        # temporarily dump
        with open(save_path+f'batch_{idx}', 'w') as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
            f.close()

    # return answers



if __name__ == '__main__':
    # print_info_delay(60)

    
    df_prompt = pd.read_csv('/usr/ninka/my_code/data/test_prompt_response_330binary.csv')
    prompt = df_prompt['text']
    labels = df_prompt['label']
    input = prompt.tolist()
    groundtruth = labels.tolist()
    
    # import pdb;pdb.set_trace()
    save_path = './result/'
    prespect_api_batch_thread(input, groundtruth, save_path)

    # with open('output.json', 'w') as json_file:  
    #     json.dump(output[0], json_file) 
    # print('output', output)