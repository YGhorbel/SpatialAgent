import boto3
import os
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_folder, bucket_name, s3_prefix=''):
    s3 = boto3.client('s3')

    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            try:
                s3.upload_file(local_path, bucket_name, s3_path)
                print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")
            except FileNotFoundError:
                print(f"The file was not found: {local_path}")
            except NoCredentialsError:
                print("Credentials not available")
            except Exception as e:
                print(f"Error uploading {local_path}: {e}")

def create_bucket_if_not_exists(bucket_name, region=None):
    s3 = boto3.client('s3', region_name=region)
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} exists.")
    except:
        print(f"Creating bucket {bucket_name}...")
        try:
            if region is None or region == "us-east-1":
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
            print(f"Bucket {bucket_name} created.")
        except Exception as e:
            print(f"Failed to create bucket: {e}")
            return False
    return True

if __name__ == "__main__":
    # Configuration
    BUCKET_NAME = 'spatial-agent-data-learner-lab'  # Change this to your bucket name
    

    # Upload Checkpoints
    print("Uploading Checkpoints...")
    upload_to_s3('distance_est/ckpt', BUCKET_NAME, 'checkpoints/distance_est')
    upload_to_s3('inside_pred/ckpt', BUCKET_NAME, 'checkpoints/inside_pred')
