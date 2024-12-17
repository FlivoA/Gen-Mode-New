# Install dependencies
!pip install transformers einops accelerate bitsandbytes

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64

# Load model and tokenizer
checkpoint = 'MBZUAI/LaMini-T5-738M'

!pip install --upgrade langchain langchain-huggingface

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float32)

from langchain_huggingface import HuggingFacePipeline

def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Test input prompt
input_prompt = "Write an article on Machine Learning"

# Generate text
model = llm_pipeline()
generated_text = model(input_prompt)
print(generated_text)

# Install and configure SageMaker dependencies
!pip uninstall -y sagemaker
!pip install sagemaker==2.168.0
!pip install packaging==21.3 --force-reinstall
!pip show packaging
!pip install protobuf==3.20.3 --force-reinstall

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import sagemaker

!pip install -U sagemaker

import json
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration
hub = {
    'HF_MODEL_ID': 'MBZUAI/LaMini-T5-738M',
    'HF_TASK': "text2text-generation",
    'device_map': 'auto',
    'torch_dtype': 'torch.float32'
}

# Create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="2.3.1"),
    env=hub,
    role=role,
)

# Deploy model to SageMaker
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    container_startup_health_check_timeout=300,
)

# Send request
response = predictor.predict({
    "inputs": "Write a short article on machine learning.",
})

print(response)

# Hyperparameters payload
payload = {
    "inputs": "Write a short article on Machine Learning",
    "parameters": {
        "do_sample": True,
        "top_p": 0.7,
        "temperature": 0.3,
        "top_k": 50,
        "max_new_tokens": 503,
        "repetition_penalty": 1.03,
    }
}

# Send request to the endpoint
response = predictor.predict(payload)
print(response)

# Inference with endpoint
ENDPOINT = "huggingface-pytorch-tgi-inference-2024-12-16-11-04-06-065"

runtime = boto3.client('runtime.sagemaker')

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT, ContentType="application/json", Body=json.dumps(payload)
)

print(response)

prediction = json.loads(response['Body'].read().decode('utf-8'))
print(prediction)

generated_text = prediction[0]['generated_text']
print(generated_text)
