## MSc Thesis Project - Adaptation of Large Language Models for the Automotive Software Engineering Domain
This repository provides the code and processed results for my Master thesis on the topic "Adaptation of Large Language Models for the Automotive Software Engineering Domain". All scripts used in the experimental pipeline are included for replicability of the results, which requires a combination of script input arguments and editing constants in the files themselves to specify datasets/paths to be used. Certain scripts should be run from the project root while others require navigating to the relevant folder. Explanations for running the scripts is provided below.

### Cloning and setup
Clone the repository:  
`git clone https://github.com/hesvo/MSc2025-ThesisProject.git`

Install requirements:  
`pip install -r requirements.txt`

Create a `.env` file with the full path to the repository root and your HuggingFace token:  
`HF_TOKEN=huggingface_token`  
`PROJECT_ROOT=/path/to/repo/root`

### Dataset repository cloning
To clone the repositories, navigate to `/repositories` and run:  
`python clone_repos.py`

Create a new folder (e.g. `/repositories/test_repos`) to separate the held-out test repositories from the training set. Simply cut-and-paste the specific repositories listed in `/repositories/test_repos.txt`.

### Data preparation
(Note: for the data preparation scripts use "`all`" as second script argument. The scripts are set up to allow processing of the test repos, but for creating the training dataset used in our experimentation full set of repositories cloned to `/repositories/all_repos` must be used. The first script argument determines which variation of the training dataset is processed.)

Ensure you are in the repository root and run the following scripts to process the repositories into the training dataset:

**Filtering**
Run (use `core` argument for "Core" (code only) dataset and `extended` for "Extended" (documentation files included) dataset):  
`python scripts/data_preparation/01_filter_repositories.py <core|extended> <all>`

**Processing (heuristic filtering, boilerplate removal, PII redaction)**
Run:  
`python scripts/data_preparation/02_process_files.py <core|extended> <all>`

**Exact and near-deduplication**
(Exact dedup) run:  
`python scripts/data_preparation/03a_exact_deduplication.py <core|extended> <all>`

(Near dedup) run:  
`python scripts/data_preparation/03b_near_deduplication.py <core|extended> <all>`

After each of these scripts has been run, the processed dataset can be found as a JSONL file in the relevant subfolder for the dataset variation ("core" or "extended")
e.g:  
`/scripts/data_preparation/03_deduplication/core/final_dataset_core.jsonl`

### Prompts
Prompts are pre-processed and stored in `/benchmark_dataset/prompts/created_prompts`. All created prompts are already available in the repository to be used for running the benchmark dataset.  
To recreate the prompts, navigate to `/benchmark_dataset/prompts` and run:  
`python create_prompts_generation.py <xl|auto> <short|long> <zero|one|three> [subset]`  
(or use `create_prompts_summarization.py`for the summarization task prompts)

(Note: when recreating prompts for the generation task, adjust the constant at the start of the `create_prompts_generation.py`script to match the template and input csv filepath depending depending on whether the file-context included prompts are being created.)

### Model training
Ensure you are in the repository root.  
Specify the `DATASET_TYPE` ("core" or "extended") in `scripts/training/config.py` (note: this file can also be used to configure further training parameters including: model_id, seed, max_sequence_length, batch_size, learning_rate, early_stopping_patience, and LoRA parameters).

To train the model (with metadata in the training dataset):  
`python scripts/training/domain_training.py`

To train the model (without metadata in the training dataset):  
`python scripts/training/domain_training_v2.py`

### Run benchmark
All our experimental results are already included in the repository under the `/results` folder.

To replicate results:  
Run the benchmark script from the repository root, with arguments specifying the task type and prompt construction (and optional argument to run benchmark subset):  
`python scripts/benchmark/run_benchmark.py <generation|summarization> <xl|auto> <short|long> <zero|one|three> [subset]`  
(note: adjust the script constants `RESULT_TYPE` ("baseline" or "adapted") and `DATASET_TYPE` ("core" or "extended") to run the desired baseline or adapted model)

Model outputs are stored in CSV files in `/results/benchmark/<baseline|adapted>/habrok`  
For further processing, create a new folder called `to_process` in `/results/benchmark/<baseline|adapted>` and copy the CSV files to this new folder.  
Then run:  
`python remove_example_problems.py three folder`  
This removes the problems that were used as few-shot examples in the prompts from the benchmark outputs, before results are processed with evaluation metrics.

### Evaluation metrics
Ensure you are in the repository root.  
Run the relevant evaluation script for the summarization or generation task, using input arguments to specify the prompt construction being evaluated, e.g.:  
`python scripts/evaluation/evaluation_generation.py <xl|auto> <short|long> <zero|one|three> [subset]`  
(note: adjust the constants `RESULTS_SUBFOLDER` and `DATASET_TYPE` to specify evaluation of the baseline or adapted model)

All evaluation results of our experimentation can be found in the `/results/evaluation` folder, organized into further subfolders specifying the base or adapted model results. Additionally, we provide further processed results such as heatmap of score differences between the base and adapted models, plots for comparing prompt engineering techniques, and statistical analysis results for the Wilcoxon signed-rank test in the folder `results/evaluation/additional_results`.
