from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="IDEA-Research/GroundingDINO",
    filename="groundingdino_swint_ogc.pth",
    subfolder="weights",
    local_dir="/data0/karan/GroundingDINO/weights",
    local_dir_use_symlinks=False
)
