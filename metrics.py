import csv
import re
import json
from pathlib import Path
from collections.abc import Sequence
from typing import List, Dict
from functools import wraps
from evaluate import EvaluationModule, combine, load

def clean_caption(caption: str) -> str:
    """
    Clean the caption by removing unwanted metadata like <|eot_id|>.
    Args:
        caption (str): The raw caption text.
    Returns:
        str: The cleaned caption.
    """
    # Remove any unwanted metadata like <|eot_id|>
    cleaned_caption = re.sub(r'<.*?>', '', caption)
    return cleaned_caption.strip()

def load_captions_from_csv(csv_file: str) -> Dict[str, str]:
    """
    Load captions from a CSV file and clean them.
    Args:
        csv_file (str): Path to the CSV file containing image filenames and captions.
    Returns:
        dict: A dictionary mapping image filenames to their corresponding cleaned captions.
    """
    captions = {}
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0]
            caption = row[1]
            cleaned_caption = clean_caption(caption)
            captions[filename] = cleaned_caption
    return captions

# Load the ground truth captions from a JSON file
def load_ground_truth(file: str | Path) -> Dict[str, str]:
    """
    Load the ground truth captions from a JSON file.
    Args:
        file (str | Path): Path to the JSON file containing the ground truth captions.
    Returns:
        dict[str, str]: A dictionary mapping image filenames to their corresponding captions.
    """
    with open(file, "r") as f:
        data: list[JSONFormat] = json.load(f)
    
    ground_truths: dict[str, str] = {}
    for item in data:
        name: str = str(item["frame_time"])
        caption: str = item["captions"].popitem()[0]
        ground_truths[f'{name}.jpg'] = caption
    return ground_truths



# Ensure that the predictions and references have matching types
def ensure_matching_types(func):
    @wraps(func)
    def wrapper(self, prediction: str | Sequence[str], reference: str | Sequence[str]) -> dict[str, float] | Sequence[float]:
        if isinstance(prediction, str):
            prediction = [prediction]
            if not isinstance(reference, str):
                raise TypeError("If prediction is a single string, reference must also be a single string.")
            reference = [reference]
        if len(prediction) != len(reference):
            raise ValueError("Prediction and reference must have the same length.")
        return func(self, prediction, reference)
    return wrapper


# Class to handle evaluation
class HuggingFaceEvaluator:
    def __init__(self, *evaluators: str) -> None:
        evaluators_list = list(evaluators)
        if "bertscore" in evaluators_list:
            evaluators_list.remove("bertscore")
            self.bertscore = load("bertscore")
        else:
            self.bertscore = None

        self.evaluator = combine(evaluators_list)

    @ensure_matching_types
    def evaluate(self, prediction: Sequence[str], reference: Sequence[str]) -> dict[str, float] | Sequence[float]:
        """
        Evaluate the prediction against the reference using Hugging Face's evaluation module.
        Args:
            prediction (str | Sequence[str]): The predicted caption(s).
            reference (str | Sequence[str]): The reference/ground-truth caption(s).
        Returns:
            dict[str, float]: The evaluation score.
        """
        if not self.bertscore:
            return self.evaluator.compute(predictions=prediction, references=reference)
        else:
            return (self.evaluator.compute(predictions=prediction, references=reference)
                    | self.bertscore.compute(predictions=prediction, references=reference, model_type="bert-base-uncased", lang="en"))

# Save evaluation results to a JSON file
def save_results_to_json(results: dict, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

# Main function to process and evaluate captions
def main():
    # Define the path to your CSV containing predicted captions
    captions_csv = '/mnt/local/karan/output/captions_gpu.csv'
    
    # Load the predicted captions
    predictions = load_captions_from_csv(captions_csv)
    
    # Define the path to your ground truth captions (assumed to be in JSON format)
    ground_truth_file = "/mnt/local/karan/BajiraoMastani/Frames.json"
    ground_truth = load_ground_truth(ground_truth_file)
    
    # Prepare the evaluator (e.g., using BLEU, ROUGE, etc.)
    evaluator = HuggingFaceEvaluator("bleu", "meteor", "rouge", "bertscore")
    
    # Match the predictions with the ground truth based on filenames
    common_files = set(ground_truth.keys()).intersection(predictions.keys())
    prediction_list = [predictions[file] for file in common_files]
    reference_list = [ground_truth[file] for file in common_files]
    
    # Evaluate and output scores
    scores = evaluator.evaluate(prediction_list, reference_list)
    print("Evaluation Results:")
    print(scores)

    # Save the evaluation results to a JSON file
    output_file = 'evaluation_results.json'
    save_results_to_json(scores, output_file)

if __name__ == "__main__":
    main()
