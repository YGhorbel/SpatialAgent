import json
import boto3
import os
import re
import ast

# --- Configuration ---
LLM_ENDPOINT_NAME = os.environ.get('LLM_ENDPOINT_NAME')
DIST_ENDPOINT_NAME = os.environ.get('DIST_ENDPOINT_NAME')
INSIDE_ENDPOINT_NAME = os.environ.get('INSIDE_ENDPOINT_NAME')
S3_BUCKET = os.environ.get('S3_BUCKET')

sagemaker_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    """
    Main entry point for API Gateway.
    Event body should contain:
    {
        "question": "...",
        "image_id": "...",
        "rle_data": [...], (Optional, if not provided, must be fetched or passed in session)
        "history": [...] (Conversation history)
    }
    """
    try:
        body = json.loads(event['body'])
        question = body.get('question')
        image_id = body.get('image_id')
        rle_data = body.get('rle_data')
        history = body.get('history', [])
        
        if not question or not image_id:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing question or image_id'})}

        # 1. Rephrase/Process Question (Simplified: Direct LLM call or use history)
        # For this agent, we loop until answer.
        
        # Initialize Agent State
        image_s3_uri = f"s3://{S3_BUCKET}/data/test/images/{image_id}" # Adjust path as needed
        
        # Add user message
        history.append({"role": "user", "content": question})
        
        # --- Conversation Loop (Simplified for Lambda - One Turn or Loop?) ---
        # Lambda has a timeout. We'll do a limited loop (e.g., 5 steps).
        
        max_steps = 5
        current_step = 0
        final_answer = None
        
        while current_step < max_steps:
            current_step += 1
            
            # 2. Call LLM
            prompt = format_prompt(history)
            llm_response = invoke_llm(prompt)
            history.append({"role": "assistant", "content": llm_response})
            
            # 3. Check for Actions
            if "<answer>" in llm_response:
                final_answer = extract_answer(llm_response)
                break
            
            if "<execute>" in llm_response:
                command = extract_command(llm_response)
                result = execute_tool(command, image_s3_uri, rle_data)
                history.append({"role": "user", "content": f"{result}"})
            else:
                # No command, no answer?
                break
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'answer': final_answer,
                'history': history
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def invoke_llm(prompt):
    # Adjust payload for your specific LLM (e.g., Llama 3)
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.1}
    }
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=LLM_ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    result = json.loads(response['Body'].read().decode())
    # Adjust parsing based on model output format
    if isinstance(result, list):
        return result[0]['generated_text']
    return result.get('generated_text', '')

def execute_tool(command, image_s3_uri, rle_data):
    # Parse command: e.g., dist(pallet_0, transporter_1)
    func_name, args = parse_command(command)
    
    if func_name == 'dist':
        return call_spatial_tool(DIST_ENDPOINT_NAME, image_s3_uri, args, rle_data)
    elif func_name == 'inside':
        return call_spatial_tool(INSIDE_ENDPOINT_NAME, image_s3_uri, args, rle_data)
    # ... implement other tools ...
    return "Tool not implemented"

def call_spatial_tool(endpoint, image_s3_uri, args, rle_data):
    # args is list of mask names: ['pallet_0', 'transporter_1']
    # Find RLEs
    mask1 = find_mask(args[0], rle_data)
    mask2 = find_mask(args[1], rle_data) # Assuming binary tools for now
    
    payload = {
        "image_s3_uri": image_s3_uri,
        "pairs": [{"mask1": mask1, "mask2": mask2}]
    }
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    result = json.loads(response['Body'].read().decode())
    return result[0]

# --- Helpers ---
def format_prompt(history):
    # Simple formatting
    text = ""
    for msg in history:
        text += f"{msg['role']}: {msg['content']}\n"
    return text

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_command(text):
    match = re.search(r"<execute>(.*?)</execute>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def parse_command(command):
    match = re.match(r"(\w+)\s*\((.*)\)", command)
    if match:
        return match.group(1), [x.strip() for x in match.group(2).split(',')]
    return None, []

def find_mask(mask_name, rle_data):
    # Logic to map 'pallet_0' to RLE from rle_data list
    # This requires the parsing logic from mask.py or simplified
    # For now, assuming rle_data is a dict keyed by mask_name or similar
    # In reality, you need to port the `parse_masks_from_conversation` logic here
    # or pass pre-processed map.
    pass 
