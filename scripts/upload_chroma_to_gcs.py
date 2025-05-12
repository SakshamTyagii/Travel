import os
import shutil
import tempfile
from google.cloud import storage

def upload_chroma_to_gcs():
    """
    Script to upload the local ChromaDB to Google Cloud Storage
    This should be run before deploying to Cloud Run
    """
    # Get the ChromaDB local path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_chroma_path = os.path.join(base_dir, "data", "chroma_db")
    
    if not os.path.exists(local_chroma_path):
        print(f"Error: ChromaDB directory not found at {local_chroma_path}")
        return
    
    # Create a temporary zip archive of the chroma_db directory
    temp_dir = tempfile.mkdtemp()
    temp_zip = os.path.join(temp_dir, "chroma_db_backup.zip")
    
    try:
        # Zip the chroma_db directory
        print(f"Compressing ChromaDB directory: {local_chroma_path}")
        shutil.make_archive(
            os.path.join(temp_dir, "chroma_db_backup"),
            'zip',
            os.path.dirname(local_chroma_path),
            os.path.basename(local_chroma_path)
        )
        
        # Upload to GCS - with error handling for permissions
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "riitii-app-chroma-db")
        print(f"Connecting to GCS bucket: {bucket_name}")
        
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            # Try to check if we have access by listing buckets
            buckets = list(storage_client.list_buckets(max_results=1))
            print(f"Successfully connected to GCS with permissions to list buckets")
        except Exception as e:
            print(f"Warning: Limited permissions detected: {str(e)}")
            print("Attempting to use service account for direct bucket access...")
        
        destination_blob_name = "chroma_db/chroma_db_backup.zip"
        blob = bucket.blob(destination_blob_name)
        
        print(f"Uploading ChromaDB to gs://{bucket_name}/{destination_blob_name}")
        blob.upload_from_filename(temp_zip)
        
        print(f"Successfully uploaded ChromaDB to GCS!")
        
        # Also upload the chroma_db directory directly
        upload_directory_to_gcs(storage_client, bucket_name, local_chroma_path, "chroma_db")
        
    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def upload_directory_to_gcs(storage_client, bucket_name, source_dir, destination_prefix):
    """Upload a directory to GCS, preserving the directory structure"""
    bucket = storage_client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create the blob path by replacing the local path prefix with GCS prefix
            relative_path = os.path.relpath(local_path, os.path.dirname(source_dir))
            blob_path = f"{destination_prefix}/{relative_path}"
            
            # Upload the file
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")

if __name__ == "__main__":
  # Set GCS bucket name environment variable before running
    if not os.environ.get("GCS_BUCKET_NAME"):
        # Default to the bucket name from .env if available
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')):
            try:
                with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'), 'r') as env_file:
                    for line in env_file:
                        if line.startswith('GCS_BUCKET_NAME='):
                            bucket_name = line.strip().split('=', 1)[1]
                            os.environ["GCS_BUCKET_NAME"] = bucket_name
                            print(f"Using bucket name from .env: {bucket_name}")
                            break
            except Exception as e:
                print(f"Error reading .env file: {e}")
                
        # If still not set, prompt user
        if not os.environ.get("GCS_BUCKET_NAME"):
            bucket_name = input("Enter your GCS bucket name: ")
            os.environ["GCS_BUCKET_NAME"] = bucket_name
        
    # Set GOOGLE_APPLICATION_CREDENTIALS environment variable if needed
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        credentials_path = input("Enter the path to your Google credentials JSON file: ")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    
    upload_chroma_to_gcs()
