# STELLAR - Search-based Testing of LLM Applications

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![OpenSBT](https://img.shields.io/badge/built_on-OpenSBT-purple.svg)](https://github.com/opensbt/opensbt-core)

<p align="center">
  <img src="../figures/approach-overview.png" alt="Architecture of STELLAR" width="400">
</p>

STELLAR is a modular search-based testing framework for evaluating LLM-based applications. It builds upon the <a href="https://www.github.com/opensbt">OpenSBT</a> infrastructure which is based on [Pymoo](https://pymoo.org/) 0.6.1.5. STELLAR provides the following core capabilities:

1. Integration of content, stylistic and perturbation features for test input generation & feature constraint handling.
2. Automated test input generation using prompting and RAG-integration.
3. Fitness evaluation leveraging LLM-based judgments alongside conventional similarity metrics (e.g., cosine similarity).
4. Search-based fitness optimization to support effectiv/efficient failure localization.

## Project Structure 

```bash
stellar/
│
├── analysis/            # Paper analysis scripts
├── configs/             # Feature config files
├── examples/            # Implementation of use cases Navi and Safety
├── judge_eval/          # Files for the judge evaluation
├── llm/                 # Main folder extending OpenSBT to support LLM Testing
├── opensbt/             # OpenSBT Base Folder
├── .env-example         # Example .env file to use cloud LLMs
├── README.md            # Project overview
├── requirements.txt     # Dependencies
├── run_tests_navi.py    # Run navi case study
└── run_tests_safety.py  # Run safety case study
```
## Installation

STELLAR requires Python to be installed and its compatibility has been tested with Python 3.11. STELLAR does **not** require GPU resources if cloud LLMs are used.

You can install dependencies via:

```bash
pip install -r requirements.txt
```

## Getting Started

To run a simplified example with a non-LLM related problem run to verify installation:

```python
python run.py -e 2
```

The configuration for LLM related experiments is done via the [config.py](./llm/config.py) as well as directly by passing arguments via flags to a corresponding function.

Make sure to provide an OpenAI/Azure OpenAI API key in [.env](./.env) to use cloud models. Note, that also local models can be used, as part of the paper experiments is using local models from [Ollama](https://ollama.com). For local models, make sure that they have been downloaded via Ollama locally, and the hardware requirements are satisfied to able to deploy the models appropriately.

To run a simplified example where an LLM is asked to provide a place recommendation:

```python
DEPLOYMENT_NAME="gpt-4o-mini" python run_tests_navi.py \
        --sut "IPA_YELP" \
        --population_size 5 \
        --n_generations 5 \
        --algorithm "nsga2" \
        --max_time "00:10:00" \
        --features_config "configs/navi_features.json"\
        --use_repair \
        --no_wandb \
        --use_rag \
        --seed 1
```

The execution should generate 25 test cases and write down all results in a folder called **results**.

For input generation, STELLAR distinguishes between style, content and perturbation features. 
The features are defined in the format as implemented in [navi_features.json](configs/navi_features.json).
You can modify theses values to see how it affects generated test inputs.

## Replication

### RQ0

To run the judge evaluation you can use the following script to collect judge results for a given set of question answer pairs. The backend LLM of the LLM-application can be directly set via the __deployment_name__ passed in the commands (here: gpt-4o-mini). 

```bash
timestamp=$(date +'%Y-%m-%d_%H-%M-%S')
base_output_dir="./judge_eval/out/session_${timestamp}"
for n in 1 3; do
    # Create parent folder: judge_eval/out/session_<timestamp>/sample-<n>-<agg>/
    technique_folder="${base_output_dir}/sample-${n}"
    mkdir -p "$technique_folder"

    for i in {1..6}; do
        # Create run subfolder
        run_folder="${technique_folder}/run${i}"
        mkdir -p "$run_folder"

        python -m judge_eval.nuanced_validation_dim \
            --models gpt-35-turbo DeepSeek-V3-0324 gpt-4o-mini gpt-4 gpt-4o gpt-5-chat mistral deepseek-v2 \
            --exp_name "sample-${n}-run${i}" \
            --dataset_path "<path to question answer pairs>" \
            --output_folder "$run_folder" \
            --n_questions 1000 \
            --n_samples $n \
            --aggregator "majority"
    done
done
```

To aggregate judge results evaluate for mulitple runs you can use the following scripts:

```bash

#!/bin/bash
BASE_DIR="<path to the runs>"
GT_CSV="<path to aggregated human annotations>"
SAVE_DIR="./judge-eval/tmp"

mkdir -p "$SAVE_DIR"

# List of sample configurations and runs
SAMPLES=("sample-1-majority" "sample-3-majority")
RUNS=("run1" "run2" "run3" "run4" "run5" "run6")

for SAMPLE in "${SAMPLES[@]}"; do
  for RUN in "${RUNS[@]}"; do
    RUN_DIR="${BASE_DIR}/${SAMPLE}/${RUN}/${SAMPLE}-${RUN}"

    CSV_PATH=$(find "$RUN_DIR" -maxdepth 1 -type f -name "*validation-repeat*.csv" | head -n 1)

    if [[ -f "$CSV_PATH" ]]; then
      OUT_DIR="${SAVE_DIR}/${SAMPLE}/${RUN}"
      mkdir -p "$OUT_DIR"
      echo "Evaluating: ${SAMPLE} - ${RUN}"
      python -m analysis.rq0.evaluate_accuracy_judge_gt \
        --csv_path "$CSV_PATH" \
        --gt_csv_path "$GT_CSV" \
        --json_output_path "${OUT_DIR}/evaluation_results.json" \
        --plot_errors_file "${OUT_DIR}/prediction_errors.png" \
        --plot_efficiency_file "${OUT_DIR}/model_efficiency.png" \
        --plot_f1_file "${OUT_DIR}/model_f1_score.png"

      echo "Done."
    else
      echo "No validation-repeat CSV found in ${RUN_DIR}"
    fi
  done
done
```

You can then run the statistical tests with:

```
bash analysis/rq0/run_statistical_test.sh
```

### RQ1

#### SafeQA

To replicate SafeQA experiments you can run the following command. As seeds, numbers between 1 and 6 have been used in the paper:

```bash
DATE=$(date +%d-%m-%Y)

# RANDOM
python run_tests_safety.py \
                --population_size 2000 \
                --n_generations 1 \
                --algorithm rs \
                --max_time "02:00:00" \
                --results_folder "/results/${DATE}/" \
                --features_config "configs/safety_features.json"\
                --seed 1

# T-wise
python run_tests_safety.py \
        --population_size 2000 \
        --n_generations 1 \
        --algorithm gs \
        --max_time "02:00:00" \
        --results_folder "/results/${DATE}/" \
        --features_config "configs/safety_features.json"\
        --seed 1

# STELLAR
python run_tests_safety.py \
        --population_size 20 \
        --n_generations 100 \
        --algorithm nsga2 \
        --max_time "02:00:00" \
        --results_folder "/results/${DATE}/" \
        --features_config "configs/safety_features.json"\
        --seed 1 \
        --use_repair
```           
#### NaviQA

To replicate NaviQA experiments you need to start for the [NaviQA](/naviqa/) application.
Then you can run the following command. As seeds numbers between 1 and 6 have been used:

```bash
DATE=$(date +%d-%m-%Y)

# RANDOM
N_VALIDATORS=1 DEPLOYMENT_NAME="gpt-4o-mini" python run_tests_navi.py \
        --sut "IPA_YELP" \
        --population_size 10000 \
        --algorithm rs \
        --max_time "03:00:00" \
        --results_folder "/results/${DATE}/" \
        --features_config "configs/navi_features.json"\
        --no_wandb \
        --use_rag \
        --seed 1

# T-wise
N_VALIDATORS=1 DEPLOYMENT_NAME="gpt-4o-mini" python run_tests_navi.py \
        --sut "IPA_YELP" \
        --population_size 10000 \
        --algorithm gs \
        --max_time "00:30:00" \
        --results_folder "/results/${DATE}/" \
        --features_config "configs/navi_features.json"\
        --no_wandb \
        --use_rag \
        --seed 1

# STELLAR
N_VALIDATORS=1 DEPLOYMENT_NAME="gpt-4o-mini" python run_tests_navi.py \
        --sut "IPA_YELP" \
        --population_size 20 \
        --n_generations 30 \
        --algorithm "nsga2" \
        --max_time "03:00:00" \
        --results_folder "/results/${DATE}/" \
        --features_config "configs/navi_features.json"\
        --use_repair \
        --no_wandb \
        --use_rag \
        --seed 1
```

### RQ2

To replicate the metric results including the diversity analysis you can run after all search runs have been completed the following scripts. It is suggested to use wandb storage to store/retrieve the experiment results. The diversity scripts can be also applied to locally stored experiments, after minor modifications in the experiment results retrieval function.

#### SafeQA

```bash
python -m analysis.rq12.get_analysis_safety
```

#### NaviQA

```bash
python -m analysis.rq12.get_analysis_navi
```

You can set the oracle threshold using __th_content=0.75__ and  __th_response=0.75__ to observe how the metrics results vary when the oracle changes.


## Customization

You can define your own custom problem as done for the Safety or Navigation case study. 
We have provided interfaces and instructions as described in [CUSTOMIZATION](CUSTOMIZATION.md).

