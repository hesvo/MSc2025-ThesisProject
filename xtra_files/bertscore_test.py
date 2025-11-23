from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
project_root_str = os.getenv("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_str)
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ['HF_HOME'] = str(HF_CACHE_DIR)

from evaluate import load
bertscore = load("bertscore")
predictions = ["Sets the gain for a specified DAC (0 or 1) on a high-voltage shield by updating its value in a cached data structure and then calling an update function to write the new configuration to the hardware, returning an error for an invalid DAC index"]
references = ["Sets the gain value for a specified DAC (0 or 1) within the HV shield device by updating a corresponding register and subsequently invoking an update function to apply the new gain, returning an error if the DAC number is invalid."]
results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=True)

print("BERTScore Results:", results)