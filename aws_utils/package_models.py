import os
import tarfile
import shutil
import boto3

def package_model(model_name, ckpt_path, inference_script_path, output_tar_path):
    print(f"Packaging {model_name}...")
    
    # Create a temporary directory for packaging
    temp_dir = f"temp_{model_name}"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Copy model checkpoint
    # We assume the user wants the best checkpoint.
    # For this script, we'll look for specific files or take the first .pth
    
    model_file = None
    if os.path.isfile(ckpt_path):
        model_file = ckpt_path
    elif os.path.isdir(ckpt_path):
        # Heuristic: find .pth file
        for f in os.listdir(ckpt_path):
            if f.endswith('.pth'):
                model_file = os.path.join(ckpt_path, f)
                break
    
    if not model_file:
        print(f"No .pth file found in {ckpt_path}")
        return False
        
    print(f"Using checkpoint: {model_file}")
    shutil.copy(model_file, os.path.join(temp_dir, 'model.pth'))
    
    # Create code directory
    code_dir = os.path.join(temp_dir, 'code')
    os.makedirs(code_dir)
    
    # Copy inference script
    shutil.copy(inference_script_path, os.path.join(code_dir, 'inference.py'))
    
    # Create requirements.txt if needed (minimal)
    with open(os.path.join(code_dir, 'requirements.txt'), 'w') as f:
        f.write("numpy\npillow\ntorch\ntorchvision\n")

    # Tar it up
    with tarfile.open(output_tar_path, "w:gz") as tar:
        tar.add(temp_dir, arcname=".")
        
    print(f"Created {output_tar_path}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    return True

def upload_to_s3(file_path, bucket_name, key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_path, bucket_name, key)
        print(f"Uploaded {file_path} to s3://{bucket_name}/{key}")
        return f"s3://{bucket_name}/{key}"
    except Exception as e:
        print(f"Failed to upload: {e}")
        return None

if __name__ == "__main__":
    BUCKET_NAME = 'spatial-agent-data-learner-lab'
    
    # Ensure bucket exists (reuse logic or assume existing from previous step)
    
    # 1. Package Distance Model
    # Adjust paths to where they are in the workspace
    dist_ckpt = 'distance_est/ckpt/epoch_5_iter_6831.pth' # Specific file
    inference_script = 'model_serving/inference.py'
    
    if package_model('distance', dist_ckpt, inference_script, 'dist_model.tar.gz'):
        upload_to_s3('dist_model.tar.gz', BUCKET_NAME, 'models/dist/model.tar.gz')
        
    # 2. Package Inside Model
    inside_ckpt = 'inside_pred/ckpt/epoch_4.pth'
    if package_model('inside', inside_ckpt, inference_script, 'inside_model.tar.gz'):
        upload_to_s3('inside_model.tar.gz', BUCKET_NAME, 'models/inside/model.tar.gz')
        
    # Cleanup local tar files
    if os.path.exists('dist_model.tar.gz'):
        os.remove('dist_model.tar.gz')
    if os.path.exists('inside_model.tar.gz'):
        os.remove('inside_model.tar.gz')
