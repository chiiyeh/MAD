# MMLU-Pro Multi-Agent LLM Experiment Runner

This project provides a Python script to run multi-agent language model experiments on the MMLU-Pro dataset. It features asynchronous API calls with retries, configurable prompt templates stored in external YAML files, and command-line arguments for experiment parameters.

## Features

* **Multi-Agent Simulation:** Simulate multiple independent agents collaborating over several rounds to answer a question.
* **Asynchronous API Calls:** Uses `asyncio` and the OpenAI Python client's async capabilities to make API calls concurrently, speeding up execution when simulating multiple agents per round.
* **Retry Mechanism:** Implements robust retries for API calls using `tenacity` to handle transient errors like rate limits or connection issues.
* **External Prompt Management:** Define and manage your initial and collaborative prompt templates in separate YAML files.
* **Jinja2 Templating:** Prompts use Jinja2 syntax, allowing dynamic injection of variables like the question, options, category, and other agents' responses.
* **Configurable Parameters:** Experiment parameters (number of samples, agents, rounds, temperature, model, data directory) are controlled via command-line arguments.
* **Structured Output:** Results are saved to a JSON file containing the full conversation history for each agent and key experiment metadata.

## Prerequisites

* An OpenAI-compatible API endpoint (like DeepInfra, OpenAI, etc.)
* An API token for your chosen endpoint.

## Setup

1.  **Clone the Repository (or save the script):**
    ```bash
    git clone https://github.com/chiiyeh/MAD.git
    cd MAD
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n mad python=3.11 -y
    conda activate mad
    ```

3.  **Install Dependencies:**
    ```bash
    conda install jinja2 openai tenacity datasets
    ```

4.  **Set up API Token:**
    The script uses the `DEEPINFRA_TOKEN` environment variable by default. If you are using DeepInfra, set this variable:
    ```bash
    export DEEPINFRA_TOKEN='YOUR_DEEPINFRA_TOKEN'
    ```
    If you are using standard OpenAI or another compatible provider, you might need to adjust the client initialization in the script (`base_url` and the environment variable name).

## Running Experiments

Execute the script from your terminal, providing the required prompt filenames and any desired parameters:

```bash
python mmlu_pro/gen_pro_mmlu_async.py --initial_prompt mmlu_pro/prompts/initial_prompt_v1.yaml --collaborative_prompt mmlu_pro/prompts/collaborative_advise_v1.yaml --num_samples 5 --data_dir ./
```

Execute the script from your terminal, providing the required prompt filenames and any desired parameters - commonsense_qa dataset:

```bash
python commonsense_qa/commonsense_qa_async.py --initial_prompt mmlu_pro/prompts/initial_prompt_v1.yaml --collaborative_prompt mmlu_pro/prompts/collaborative_advise_v1.yaml --num_samples 5 --data_dir ./
```

Execute the script from your terminal for evaluation, providing the output file name generated from the previous experiment - commonsense_qa dataset:

```bash
python commonsense_qa/eval_commonsense_qa.py --input_file commonsense_qa_3agents_2rounds_0.7temp_Qwen_Qwen2.5-7B-Instruct_initial_cot_detailed_v1_collaborative_advise_v1.json
```
