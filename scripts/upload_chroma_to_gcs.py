import os
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def upload_directory_to_gcs(source_dir, gcs_path):
    """Upload a local directory to Google Cloud Storage."""
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist!")
        return False
        
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Get the relative path from the source directory
                relative_path = os.path.relpath(local_path, source_dir)
                gcs_object_name = os.path.join(gcs_path, relative_path).replace("\\", "/")
                
                blob = bucket.blob(gcs_object_name)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{GCS_BUCKET_NAME}/{gcs_object_name}")
        
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        return False

if __name__ == "__main__":
    # Path to the local ChromaDB directory
    chroma_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
    
    # Upload to GCS
    success = upload_directory_to_gcs(chroma_db_path, "chroma_db")
    
    if success:
        print(f"Successfully uploaded ChromaDB to GCS bucket {GCS_BUCKET_NAME}")
    else:
        print("Failed to upload ChromaDB to GCS")
