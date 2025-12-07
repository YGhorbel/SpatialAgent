import boto3
import time
import sagemaker
from sagemaker.pytorch import PyTorchModel

def get_lab_role():
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    # Standard Learner Lab Role Name
    role_arn = f"arn:aws:iam::{account_id}:role/LabRole"
    return role_arn

def deploy_custom_model(model_name, s3_model_uri, role_arn, instance_type='ml.m5.large'):
    print(f"Deploying {model_name} from {s3_model_uri}...")
    
    sm_client = boto3.client('sagemaker')
    
    # Check if endpoint already exists
    endpoint_name = f"{model_name}-endpoint"
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} already exists. Skipping.")
        return endpoint_name
    except:
        pass

    # Create SageMaker Model
    # We use the generic Model class or PyTorchModel
    # Since we packaged it ourselves with inference.py, we can use PyTorchModel
    
    model = PyTorchModel(
        model_data=s3_model_uri,
        role=role_arn,
        entry_point='inference.py', # Inside the tar.gz code/ directory
        source_dir=None, # Already packaged
        framework_version='2.0',
        py_version='py310',
        name=f"{model_name}-model",
        env={'MODEL_TYPE': model_name} # 'distance' or 'inside'
    )
    
    # Deploy
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    print(f"Deployed {endpoint_name}")
    return endpoint_name

if __name__ == "__main__":
    BUCKET_NAME = 'spatial-agent-data-learner-lab'
    ROLE_ARN = get_lab_role()
    print(f"Using Role: {ROLE_ARN}")
    
    # Deploy Distance Model
    dist_uri = f"s3://{BUCKET_NAME}/models/dist/model.tar.gz"
    deploy_custom_model('distance', dist_uri, ROLE_ARN)
    
    # Deploy Inside Model
    inside_uri = f"s3://{BUCKET_NAME}/models/inside/model.tar.gz"
    deploy_custom_model('inside', inside_uri, ROLE_ARN)
    
    print("Deployment initiated. This may take several minutes.")
