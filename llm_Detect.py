import os
from openai import AzureOpenAI

endpoint = "YOUR ENDPOINT"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = "YOUR KEY"
api_version = "2024-12-01-preview"
metaprompt = f"""Carefully analyze the given MAS (Multi-Agent System) execution trace. Determine whether the execution is correct. Output your answer strictly in the following JSON format:{{"verdict": "CORRECT"}} or {{"verdict": "ERROR"}}"""
def format_sample(sample):
    trace = sample['trace']
    annotation_str = ', '.join([f"{k}: {v}" for k, v in sample['mast_annotation'].items()])
    return (
        f"trace.trajectory: {trace['trajectory']}\n"
        "===================="
    )
def get_respond(meta_prompt, user_prompt):
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    if len(user_prompt)>100000:
        user_prompt = user_prompt[:100000]
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": meta_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )

    return response.choices[0].message.content
import json

import json
import re

def extract_verdict(text):
    
    cleaned_text = re.sub(r"^```(?:json)?\n|\n```$", "", text.strip())
    try:
        parsed = json.loads(cleaned_text)
        return parsed.get("verdict", None)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

from bot_pipeline import BoT, FailureDetect
import argparse
from bert_classify import ChunkedTextClassifier
parser = argparse.ArgumentParser(description='Use of argparse')

parser.add_argument('--llm_model',type=str,default='gpt-4o-mini',help='Model id of LLMs')
parser.add_argument('--embedding_model',type=str,default='text-embedding-3-large',help='Model id of embedding model')
parser.add_argument('--api_key',type=str,default="YOUR KEY", help='The api key of user')
parser.add_argument('--base_url',type=str,default="YOUR URL",help='we also support Open AI-like chat/embeddings APIs')
parser.add_argument('--rag_dir',type=str,default='./math',help='The path to save the meta buffer')

args = parser.parse_args()

llm_model = args.llm_model
embedding_model = args.embedding_model
api_key = "YOUR KEY"
base_url = "YOUR URL"
rag_dir = args.rag_dir



import random

def build_labeled_dataset_with_labels(dataset, trian_indices_right, trian_indices_error):
    labeled_data = []
    labels = []

    for idx in trian_indices_right:
        sample = dataset[idx].copy()
        sample["label"] = 0
        labeled_data.append(sample)
        labels.append(0)

    for idx in trian_indices_error:
        sample = dataset[idx].copy()
        sample["label"] = 1
        labeled_data.append(sample)
        labels.append(1)

    combined = list(zip(labeled_data, labels))
    random.shuffle(combined)
    labeled_data, labels = zip(*combined)

    return list(labeled_data), list(labels)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
log_history = {
    "sample": [],
    "result": [],
}

labeled_dataset, true_labels = build_labeled_dataset_with_labels(data, trian_indices_right[100:200], trian_indices_error[100:200])
pred_labels = []
for i,d in enumerate(labeled_dataset):
    d = format_sample(d)
    ans = get_respond(meta_prompt = metaprompt, user_prompt=d)
    pred_text = extract_verdict(ans)
    if pred_text =="ERROR":
        pred = 1
    elif pred_text =="CORRECT":
        pred = 0
    else:
        pass
    pred_labels.append(pred)
    log_history["sample"].append(i)
    log_history["result"].append(ans)
import pandas as pd

df = pd.DataFrame(log_history)
df.to_csv("training_log_llm.csv", index=False)
print("Accuracy:", accuracy_score(true_labels, pred_labels))
print("Precision:", precision_score(true_labels, pred_labels))
print("Recall:", recall_score(true_labels, pred_labels))
print("F1 Score:", f1_score(true_labels, pred_labels))


print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=["Right (0)", "Error (1)"]))

