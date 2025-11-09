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

if not boto3.client('sts').get_caller_identity().get('Account'):
    raise EnvironmentError(
        "AWS credentials not found. Please configure your credentials safely:\n"
        "1. Use AWS CLI: aws configure\n"
        "2. Use environment variables\n"
        "3. Use .env file (do not commit this file)\n"
    )

df_politifact = pd.read_csv("politifact_kaggle_dataset_sampled.csv")

df_politifact['mistral_verdict'] = ""
df_politifact['mistral_reason'] = ""


model_id = "mistral.mistral-7b-instruct-v0:2"


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
        f"Here are some examples for each label:\n"
        f"Statement: China and Singapore impose the death penalty on drug dealers. Date: 2018-10-03. Label: true \n"
        f"Statement: Forty percent of people in this country, illegally, are overstaying visas. Date: 2015-07-22. Label: mostly true \n"
        f"Statement: Cincinnati runs 100 percent on clean energy. Date: 2016-06-27. Label: half true \n"
        f"Statement: The United States is imprisoning young people who are smoking marijuana. Date:2015-10-13. Label: mostly false \n"
        f"Statement: Most young Americans right now are not covered by health insurance. Date: 2014-11-03. Label: false \n"
        f"Statement: The heroin and fentanyl epidemic is growing because we don't have a wall. Date:2018-01-30. Label: implausible."
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
        f"Here are some examples for each label:\n"
        f"Statement: China and Singapore impose the death penalty on drug dealers. Source: Donald Trump. Date: 2018-10-03. Label: true \n"
        f"Statement: Forty percent of people in this country, illegally, are overstaying visas. Source: Marco Rubio. Date: 2015-07-22. Label: mostly true \n"
        f"Statement: Cincinnati runs 100 percent on clean energy. Source: Hillary Clinton. Date: 2016-06-27. Label: half true \n"
        f"Statement: The United States is imprisoning young people who are smoking marijuana. Source: Bernie Sanders. Date:2015-10-13. Label: mostly false \n"
        f"Statement: Most young Americans right now are not covered by health insurance. Source: Barack Obama. Date: 2014-11-03. Label: false \n"
        f"Statement: The heroin and fentanyl epidemic is growing because we don't have a wall. Source: Ann Coulter. Date:2018-01-30. Label: implausible."
    )


for i, row in df_politifact.iterrows():
    # input_prompt = prompt(row['statement'], row['statement_date'])
    input_prompt = source_prompt(
        row['statement'], row['statement_date'], row['politician_name'])

    body = {
        "prompt": input_prompt,
        "max_tokens": 500,  # accomodate for reasoning
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

        # print(json.dumps(result, indent=2)) # Debugging only

        json_answer = result['outputs'][0]['text'].strip()
        match = re.search(r'(true|mostly true|half true|mostly false|false|implausible)',
                          json_answer, re.IGNORECASE)
        answer = match.group(1).lower() if match else json_answer

        print(
            f"Source: {row['politician_name']}\n"
            f"Statement {i+1}: {row['statement']},\nDate: {row['statement_date']},\n"
            f"Statement {i+1}: Label:{answer}\nReason:{json_answer}\n\n"
        )
        # print(f"{json_answer}\n\n") # Debugging only
        df_politifact.at[i, 'mistral_verdict'] = answer
        df_politifact.at[i,
                         'mistral_reason'] = result['outputs'][0]['text'].strip()

        time.sleep(10)  # avoid throttling
    except Exception as e:
        print(f"Error on row{i}: {e}")
        if "ThrottlingException" in str(e):
            wait = random.uniform(10, 20)
            print(f"Throttled, waiting for {wait:.1f} seconds.")
            time.sleep(wait)
        df_politifact.at[i, 'mistral_verdict'] = "error"
        df_politifact.at[i, 'mistral_reason'] = "error"

df_politifact.to_csv(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs_src.csv", index=False)
print("Labelling complete.")
