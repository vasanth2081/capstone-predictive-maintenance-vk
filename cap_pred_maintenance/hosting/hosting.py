from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="cap_pred_maintenance/deployment",     # the local folder containing your file
    repo_id="v-vasanth2009/capstone-pred-main-vk-12052026",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
