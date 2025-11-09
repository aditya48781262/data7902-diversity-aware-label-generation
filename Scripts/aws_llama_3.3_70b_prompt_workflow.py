import boto3
import json
import pandas as pd
import time
import re
import random
from dotenv import load_dotenv
import os

load_dotenv()

client = boto3.client(
    'bedrock-runtime',
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

df_politifact = pd.read_csv("politifact_kaggle_dataset_sampled.csv")

df_politifact['llama_verdict'] = ""
df_politifact['llama_reason'] = ""

model_id = "meta.llama3-3-70b-instruct-v1:0"


def prompt(statement, date):
    return (
        f"You are a fact checking expert."
        f"Fact check the following statement, also considering its recorded date.\n\n"
        f"Date of statement: {date}\n"
        f"Statement: \"{statement}\"\n\n"
        f"You must label this statement based on the following criteria:"
        f"true: The statement is accurate, supported by facts and no significant information is missing."
        f"mostly true: The statement is accurate but needs clarification or additional information."
        f"half true: The statement is partially accurate but omits important details or takes things out of context."
        f"mostly false: The statement contains an element of truth but ignores critical facts that would give a different impression."
        f"false: The statement is not accurate, with no factual support or misleading information"
        f"implausible: The statement is completely inaccurate, makes outlandish claims, or is totally detached from reality."
        f"Think step by step, considering the provided criteria before labelling the statement."
    )


def source_prompt(statement, date, source):
    return (
        f"You are a fact checking expert."
        f"Fact check the following statement, also considering its recorded date and source.\n\n"
        f"Date of statement: {date}\n"
        f"Statement: \"{statement}\"\n"
        f"Source: {source}\n\n"
        f"You must label this statement based on the following criteria:"
        f"true: The statement is accurate, supported by facts and no significant information is missing.\n"
        f"mostly true: The statement is accurate but needs clarification or additional information.\n"
        f"half true: The statement is partially accurate but omits important details or takes things out of context.\n"
        f"mostly false: The statement contains an element of truth but ignores critical facts that would give a different impression.\n"
        f"false: The statement is not accurate, with no factual support or misleading information.\n"
        f"implausible: The statement is completely inaccurate, makes outlandish claims, or is totally detached from reality.\n"
        f"Think step by step, considering the provided criteria before labelling the statement."
    )


for i, row in df_politifact.iterrows():
    # input_prompt = prompt(row['statement'], row['statement_date'])
    input_prompt = source_prompt(
        row['statement'], row['statement_date'], row['politician_name'])

    body = {
        "prompt": input_prompt,
        "max_gen_len": 500,  # short labels
        "temperature": 0.0,  # deterministic output required for data labelling
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        result = json.load(response["body"])

        # print(json.dumps(result, indent=2))

        json_answer = result['generation'].strip()
        match = re.search(r'(true|mostly true|half true|mostly false|false|implausible)',
                          json_answer, re.IGNORECASE)
        answer = match.group(1).lower() if match else json_answer

        print(
            f"Source: {row['politician_name']}\n"
            f"Statement {i+1}: {row['statement']},\nDate: {row['statement_date']},\n"
            f"Statement {i+1}: Label:{answer}\nReason:{json_answer}\n\n"
        )

        df_politifact.at[i, 'llama_verdict'] = answer
        df_politifact.at[i,
                         'llama_reason'] = result['generation'].strip()

        time.sleep(10)  # avoid throttling
    except Exception as e:
        print(f"Error on row{i}: {e}")
        if "ThrottlingException" in str(e):
            wait = random.uniform(15, 25)
            print(f"Throttled, waiting for {wait:.1f} seconds.")
            time.sleep(wait)
        df_politifact.at[i, 'llama_verdict'] = "error"

df_politifact.to_csv(
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs_src.csv", index=False)
print("Labelling complete.")
