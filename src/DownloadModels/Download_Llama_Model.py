import os

from huggingface_hub import snapshot_download
from huggingface_hub import login

# Log into huggingface
def download_llama_model(download_model_path, hf_access_token, model_folder = 'llama',
                         llama_model_repo_id='meta-llama/Llama-2-13b-chat-hf'):
    login(token=hf_access_token)

    # Download LLaMA Model to Local Folder
    snapshot_download(repo_id = llama_model_repo_id,  cache_dir= os.path.join(download_model_path, model_folder))


