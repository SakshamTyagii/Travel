import os
import subprocess
import json
import sys
import webbrowser

def run_command(command, description):
    """Run a command with proper output handling"""
    print(f"\n[RUNNING] {description}...")
    print(f"Command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Success: {description}")
        return result.stdout
    else:
        print(f"❌ Error: {description} failed")
        print(f"Error details: {result.stderr}")
        return None

def check_gcloud_installed():
    """Check if gcloud is installed"""
    try:
        subprocess.run(['gcloud', '--version'], check=True, capture_output=True)
        return True
    except Exception:
        print("❌ Error: Google Cloud SDK (gcloud) is not installed or not in PATH.")
        print("Please install it from: https://cloud.google.com/sdk/docs/install")
        return False

def load_env_file():
    """Load environment variables from .env file"""
    env_vars = {}
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars

def deploy_app():
    """Deploy the application to Google Cloud Run"""
    # Check prerequisites
    if not check_gcloud_installed():
        return
    
    # Load environment variables
    env_vars = load_env_file()
    gemini_api_key = env_vars.get('GEMINI_API_KEY')
    gcs_bucket_name = env_vars.get('GCS_BUCKET_NAME')
    
    if not gemini_api_key:
        print("⚠️ Warning: GEMINI_API_KEY not found in .env file")
        gemini_api_key = input("Please enter your Gemini API key: ")
    
    if not gcs_bucket_name:
        print("⚠️ Warning: GCS_BUCKET_NAME not found in .env file")
        gcs_bucket_name = input("Please enter your GCS bucket name: ")
    
    # Get project information
    project_id = run_command("gcloud config get-value project", "Get current project ID")
    if not project_id:
        project_id = input("Enter your Google Cloud project ID: ")
        run_command(f"gcloud config set project {project_id}", "Set project ID")
    
    project_id = project_id.strip()
    print(f"Using Google Cloud Project: {project_id}")
    
    # Get service account key or create one if needed
    key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'service-account-key.json')
    
    if not os.path.exists(key_path):
        print("\n[INFO] No service account key found. You have three options:")
        print("1. Download it manually from the Google Cloud Console")
        print("2. Create a new service account key (requires admin privileges)")
        print("3. Use application default credentials (if you've run 'gcloud auth application-default login')")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == "1":
            print("\n[INFO] Please download the key manually:")
            print("1. Go to: https://console.cloud.google.com/iam-admin/serviceaccounts")
            print("2. Find your service account (cloud-run-chroma-access)")
            print("3. Click on the 'Keys' tab")
            print("4. Create and download a new JSON key")
            print("5. Save it as 'service-account-key.json' in your project root directory")
            
            webbrowser.open("https://console.cloud.google.com/iam-admin/serviceaccounts")
            input("Press Enter after you've downloaded the key...")
            
            if not os.path.exists(key_path):
                key_path = input("Enter the path to your downloaded key file: ")
        
        elif choice == "2":
            sa_name = "cloud-run-chroma-access"
            sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"
            
            # Check if service account already exists
            sa_exists = run_command(f"gcloud iam service-accounts describe {sa_email}", "Check if service account exists")
            
            if not sa_exists:
                # Create service account
                run_command(
                    f'gcloud iam service-accounts create {sa_name} --display-name="Delhi Travel Assistant"',
                    "Create service account"
                )
                
                # Grant roles
                run_command(
                    f'gcloud projects add-iam-policy-binding {project_id} --member="serviceAccount:{sa_email}" --role="roles/storage.admin"',
                    "Grant Storage Admin role"
                )
                run_command(
                    f'gcloud projects add-iam-policy-binding {project_id} --member="serviceAccount:{sa_email}" --role="roles/run.admin"',
                    "Grant Cloud Run Admin role"
                )
            
            # Create and download key
            run_command(
                f'gcloud iam service-accounts keys create {key_path} --iam-account={sa_email}',
                "Create and download service account key"
            )
        
        elif choice == "3":
            print("Using application default credentials...")
            run_command("gcloud auth application-default login", "Log in with application default credentials")
            key_path = ""
    
    # If we have a key file, set the environment variable
    if os.path.exists(key_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
        print(f"✅ Using service account key: {key_path}")
    
    # Upload ChromaDB to Cloud Storage
    os.environ['GCS_BUCKET_NAME'] = gcs_bucket_name
    upload_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload_chroma_to_gcs.py')
    
    print("\n[INFO] Would you like to upload your ChromaDB to Cloud Storage?")
    upload_choice = input("This is required the first time or if your data has changed (y/n): ").lower()
    
    if upload_choice == 'y' or upload_choice == 'yes':
        print("\n[RUNNING] Uploading ChromaDB to Cloud Storage...")
        subprocess.run([sys.executable, upload_script], env=os.environ)
    
    # Update cloudbuild.yaml with correct environment variables
    cloudbuild_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cloudbuild.yaml')
    
    # Deploy using Cloud Build
    print("\n[INFO] Ready to deploy to Cloud Run!")
    deploy_choice = input("Do you want to proceed with deployment? (y/n): ").lower()
    
    if deploy_choice == 'y' or deploy_choice == 'yes':
        deploy_cmd = f"gcloud builds submit --substitutions=_GEMINI_API_KEY={gemini_api_key},_GCS_BUCKET_NAME={gcs_bucket_name}"
        result = run_command(deploy_cmd, "Deploy to Cloud Run")
        
        if result:
            print("\n🎉 Deployment successful!")
            print("\nYour application should be available at:")
            run_command("gcloud run services list --platform managed", "Get service URL")
    
    print("\nDeployment process completed.")

if __name__ == "__main__":
    deploy_app()
