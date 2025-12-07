import boto3
import json
import time
import zipfile
import io
import os

def get_lab_role():
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    return f"arn:aws:iam::{account_id}:role/LabRole"

def create_lambda_package(source_file):
    # Create a zip file in memory
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_file, arcname="lambda_function.py")
    return mem_zip.getvalue()

def deploy_lambda(function_name, role_arn, env_vars):
    lambda_client = boto3.client('lambda')
    
    # Check if exists
    try:
        lambda_client.get_function(FunctionName=function_name)
        print(f"Function {function_name} already exists. Updating code...")
        # Update code
        zip_content = create_lambda_package('agent/lambda_function.py')
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content
        )
        # Update config
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment={'Variables': env_vars},
            Timeout=60
        )
        return
    except lambda_client.exceptions.ResourceNotFoundException:
        pass

    print(f"Creating function {function_name}...")
    zip_content = create_lambda_package('agent/lambda_function.py')
    
    lambda_client.create_function(
        FunctionName=function_name,
        Runtime='python3.10',
        Role=role_arn,
        Handler='lambda_function.lambda_handler',
        Code={'ZipFile': zip_content},
        Timeout=60,
        Environment={'Variables': env_vars},
        # Layers=['arn:aws:lambda:us-east-1:336392948345:layer:AWSLambda-Python310-SciPy1x:1'] # Example ARN, might vary
    )
    print(f"Function {function_name} created.")

def deploy_api_gateway(function_name):
    # Simplified HTTP API creation
    apigateway = boto3.client('apigatewayv2')
    lambda_client = boto3.client('lambda')
    
    api_name = f"{function_name}-API"
    
    # Check if API exists (naive check)
    apis = apigateway.get_apis()
    for item in apis['Items']:
        if item['Name'] == api_name:
            print(f"API {api_name} already exists. URL: {item['ApiEndpoint']}")
            return item['ApiEndpoint']

    print(f"Creating API {api_name}...")
    api = apigateway.create_api(
        Name=api_name,
        ProtocolType='HTTP',
        Target=f"arn:aws:lambda:us-east-1:{boto3.client('sts').get_caller_identity()['Account']}:function:{function_name}"
    )
    
    api_id = api['ApiId']
    api_endpoint = api['ApiEndpoint']
    
    # Add permission for API Gateway to invoke Lambda
    try:
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId='apigateway-invoke',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=f"arn:aws:execute-api:us-east-1:{boto3.client('sts').get_caller_identity()['Account']}:{api_id}/*"
        )
    except:
        pass # Permission might already exist

    print(f"API Created. URL: {api_endpoint}")
    return api_endpoint

if __name__ == "__main__":
    ROLE_ARN = get_lab_role()
    
    ENV_VARS = {
        'LLM_ENDPOINT_NAME': 'jumpstart-dft-meta-textgeneration-llama-3-8b', # Placeholder
        'DIST_ENDPOINT_NAME': 'distance-endpoint',
        'INSIDE_ENDPOINT_NAME': 'inside-endpoint',
        'S3_BUCKET': 'spatial-agent-data-learner-lab'
    }
    
    deploy_lambda('SpatialAgentFunction', ROLE_ARN, ENV_VARS)
    deploy_api_gateway('SpatialAgentFunction')
