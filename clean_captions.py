import re
import pandas as pd

def clean_caption_text(text: str) -> str:
    # Remove specific tags like <|eot_id|>, end_header_id|>, and similar markup
    return re.sub(r"<\|.*?\|>|end_header_id\|>", "", text).strip()

def clean_caption_file(input_path: str, output_path: str) -> None:
    # Read the file using pandas
    df = pd.read_csv(input_path, sep=",", header=None, names=["image", "caption"])
    
    # Clean each caption
    df["caption"] = df["caption"].apply(clean_caption_text)
    
    # Save the cleaned version
    df.to_csv(output_path, sep=",", index=False, header=False)


clean_caption_file("/data0/karan/output_capts_llama3/captions_GSAM_ojpspec.csv", "/data0/karan/output_capts_llama3/captions_GSAM_objspecific_cleaned.csv")
