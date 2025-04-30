import os
import openai
# Use AsyncOpenAI for async operations
from openai import AsyncOpenAI
import asyncio
import tenacity
import json
import re
import random
import time
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Tuple, Any, Optional

# Import libraries for prompt management and argument parsing
import yaml
from jinja2 import Environment, select_autoescape, Template
import argparse

# Assuming eval_mmlu.py exists and has a parse_answer function
from eval_mmlu import parse_answer

random.seed(12345)

# --- Prompt Management Setup ---

# Configure Jinja2 environment
env = Environment(
    autoescape=select_autoescape(["html", "xml"]) # Autoescape isn't strictly needed for text prompts
)


def load_prompt_template_data(filepath: Path) -> Dict:
    """Loads a prompt definition from a YAML file and returns its content."""
    try:
        with filepath.open('r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
        if not isinstance(prompt_data, dict) or 'name' not in prompt_data or 'template' not in prompt_data:
            raise ValueError(f"Prompt file {filepath} must be a dictionary containing 'name' and 'template' keys.")
        return prompt_data
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
        raise
    except ValueError as e:
         print(f"Error in prompt file format {filepath}: {e}")
         raise

# --- End Prompt Management Setup ---


# Define default max_tokens for clarity
DEFAULT_MAX_TOKENS = 3000 # Increased max tokens, might need adjustment based on model and prompt verbosity

#### CHAT UTILITIES ####

# construct_message is no longer used in the new phased flow


def construct_assistant_message(completion_text: str) -> Dict:
    """Constructs an assistant message dictionary from completion text."""
    return {"role": "assistant", "content": completion_text}


# --- Async chat function with retry ---

@tenacity.retry(
    wait=tenacity.wait_random_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(8),
    retry=tenacity.retry_if_exception_type((
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APIStatusError,
        asyncio.TimeoutError # Add TimeoutError as potentially retriable
    )),
    before_sleep=lambda retry_state: print(f"Retrying API call (attempt {retry_state.attempt_number}/{retry_state.attempt_number + retry_state.outcome.failures}). Waiting {retry_state.next_action.sleep:.2f}s after error: {retry_state.outcome.exception()}")
)
async def async_chat_with_gpt(
    client: AsyncOpenAI,
    messages: List[Dict],
    model: str,
    temperature: float,
    n: int = 1,
    stop: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Tuple[List[str], int]:
    """
    Get completions from GPT model asynchronously with retries.
    Returns a tuple: ([completion_text1, ...], total_tokens_used)
    """
    try:
        # Ensure messages is always a list, even with a single message
        if not isinstance(messages, list):
             messages = [messages]

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
        )
        tokens_used = response.usage.total_tokens if response.usage else 0
        return [choice.message.content for choice in response.choices], tokens_used
    except Exception as e:
        print(f"API call failed: {e}")
        raise e # Re-raise for tenacity or calling code


#### DATASET ####
# Keep dataset loading and processing functions
def load_mmlu_pro(data_dir: str):
    dataset = load_dataset("parquet", data_files={'test': data_dir+"/test-00000-of-00001.parquet", 'validation': data_dir+"/validation-00000-of-00001.parquet"})
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res

# format_example is needed again to prepare the question string for templates
def format_example(question: str, options: List[str]) -> str:
    """Formats a question and its options into a string."""
    example = f"Question: {question}\nOptions:\n"
    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # Increased map size
    for i, opt in enumerate(options):
        example += f"{choice_map[i]}. {opt}\n"
    return example


# --- Async Main Execution ---
async def main(
    initial_prompt_filepath: Path,
    critic_prompt_filepath: Path, # New: Critic prompt file path
    synthesizer_prompt_filepath: Path, # New: Synthesizer prompt file path
    num_samples: int,
    agents: int, # Number of initial agents
    # rounds: int, # Removed
    temperature: float,
    model: str,
    rootdir: str,
    # resume: bool, # Removed for this flow structure
):
    """
    Main asynchronous function to run multi-agent (initial -> critic -> synthesis)
    experiments. Starts fresh each time.
    """
    response_dict: Dict[str, Any] = {}

    # --- Load Prompt Templates ---
    print(f"Loading prompt templates: {initial_prompt_filepath}, {critic_prompt_filepath}, {synthesizer_prompt_filepath}")
    try:
        initial_prompt_data = load_prompt_template_data(initial_prompt_filepath)
        critic_prompt_data = load_prompt_template_data(critic_prompt_filepath)
        synthesizer_prompt_data = load_prompt_template_data(synthesizer_prompt_filepath)

        initial_template = env.from_string(initial_prompt_data['template'])
        critic_template = env.from_string(critic_prompt_data['template'])
        synthesizer_template = env.from_string(synthesizer_prompt_data['template'])


        print(f"Prompt templates loaded successfully:")
        print(f" - Initial: '{initial_prompt_data['name']}'")
        print(f" - Critic: '{critic_prompt_data['name']}'")
        print(f" - Synthesizer: '{synthesizer_prompt_data['name']}'")

    except Exception as e:
        print(f"Failed to load prompt templates. Exiting. Error: {e}")
        raise e # Re-raise to stop execution

    # --- Initialize API Client ---
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPINFRA_TOKEN", ""),
        base_url="https://api.deepinfra.com/v1/openai"
    )
    if not client.api_key:
        print("Warning: DEEPINFRA_TOKEN environment variable is not set.")
        print("API calls may fail.")

    # --- Load and Prepare Question Data ---
    # Load and preprocess data (synchronous)
    print("Loading dataset...")
    try:
        test_df, dev_df = load_mmlu_pro(os.path.join(rootdir, "dataset/mmlu_pro"))
        subjects = list(test_df.keys())
        print(f"Loaded subjects: {subjects}")
        if not subjects:
            print("Error: No subjects found in the dataset. Check data_dir.")
            return # Exit if no data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return # Exit if data loading fails


    # --- Process Questions (New Phased Flow) ---
    for i in tqdm(range(num_samples), desc="Processing questions"):
        # Select a random question
        subject = random.choice(subjects)
        df = test_df[subject]
        if not df:
            print(f"Warning: No questions found for subject {subject}. Skipping.")
            continue

        # Ensure there's at least one question in the subject's dataframe
        if len(df) == 0:
            print(f"Warning: Subject {subject} is empty. Skipping.")
            continue

        idx = random.randint(0, len(df) - 1)
        single_question = df[idx]

        # Extract question details
        q_id = single_question.get("question_id", f"{subject}_{idx}")
        answer = single_question["answer"]
        category = single_question["category"]
        question_text = single_question["question"]
        options = single_question["options"]

        question_with_options = format_example(question_text, options)

        if q_id in response_dict:
            print(f"Question ID {q_id} already processed. Skipping.")
            continue

        print(f"\n--- Question {i+1}/{num_samples} (ID: {q_id}, Subject: {category}) ---")

        initial_solution_messages: List[Dict] = []
        critic_message: Optional[Dict] = None
        synthesizer_message: Optional[Dict] = None

        # --- Phase 1: Initial Solutions from Agents ---
        print(" Running Phase 1: Initial Agents")
        initial_tasks = []
        initial_prompt_content = initial_template.render({
            "category": category,
            "question": question_with_options, # Pass raw text
        })

        for agent_idx in range(agents):
            # Each agent starts with the initial prompt
            messages = [{"role": "user", "content": initial_prompt_content}]
            task = async_chat_with_gpt(
                client,
                messages,
                model=model,
                temperature=temperature,
                max_tokens=DEFAULT_MAX_TOKENS
            )
            initial_tasks.append(task)

        try:
            initial_results = await asyncio.gather(*initial_tasks)
            # Store initial agent assistant messages
            initial_solution_messages = [
                construct_assistant_message(res[0][0]) # Assuming n=1, get the first completion text
                for res in initial_results if res and res[0]
            ]
            if len(initial_solution_messages) != agents:
                 print(f"Warning: Only {len(initial_solution_messages)} initial solutions received for QID {q_id} (expected {agents}).")

        except tenacity.RetryError as e:
            print(f"Error: Initial agents failed after multiple retries for QID {q_id}. Skipping phases 2 and 3.")
            initial_solution_messages = [] # Ensure list is empty if phase failed
        except Exception as e:
            print(f"An unexpected error occurred during initial agents phase for QID {q_id}: {e}")
            initial_solution_messages = [] # Ensure list is empty if phase failed
            pass # Continue to next question


        # --- Phase 2: Critic Agent ---
        if initial_solution_messages: # Only run if initial solutions were successful
            print(" Running Phase 2: Critic Agent")
            try:
                critic_prompt_content = critic_template.render({
                    "question": question_with_options,
                    "initial_solution_messages": initial_solution_messages # Pass list of initial assistant messages
                })
                critic_messages = [{"role": "user", "content": critic_prompt_content}]

                critic_completion, _ = await async_chat_with_gpt(
                    client,
                    critic_messages,
                    model=model,
                    temperature=temperature, # Can use different temperature for critic
                    max_tokens=DEFAULT_MAX_TOKENS
                )
                if critic_completion:
                     critic_message = construct_assistant_message(critic_completion[0])
                else:
                     print(f"Warning: Critic agent returned empty response for QID {q_id}.")

            except tenacity.RetryError as e:
                print(f"Error: Critic agent failed after multiple retries for QID {q_id}. Skipping phase 3.")
                critic_message = None
            except Exception as e:
                print(f"An unexpected error occurred during critic phase for QID {q_id}: {e}")
                critic_message = None
                pass # Continue to next question or phase


        # --- Phase 3: Synthesizer Agent ---
        if critic_message: # Only run if critic provided feedback
             print(" Running Phase 3: Synthesizer Agent")
             try:
                 synthesizer_prompt_content = synthesizer_template.render({
                     "question": question_with_options,
                     "initial_solution_messages": initial_solution_messages,
                     "critique_message": critic_message # Pass critic's assistant message
                 })

                 synthesizer_messages = [{"role": "user", "content": synthesizer_prompt_content}]

                 synthesizer_completion, _ = await async_chat_with_gpt(
                     client,
                     synthesizer_messages,
                     model=model,
                     temperature=temperature, # Can use different temperature for synthesizer
                     max_tokens=DEFAULT_MAX_TOKENS
                 )
                 if synthesizer_completion:
                     synthesizer_message = construct_assistant_message(synthesizer_completion[0])
                 else:
                     print(f"Warning: Synthesizer agent returned empty response for QID {q_id}.")

             except tenacity.RetryError as e:
                 print(f"Error: Synthesizer agent failed after multiple retries for QID {q_id}.")
                 synthesizer_message = None
             except Exception as e:
                 print(f"An unexpected error occurred during synthesizer phase for QID {q_id}: {e}")
                 synthesizer_message = None
                 pass # Continue to next question


        # --- Store Results ---
        # Store the results of all phases for this question
        response_dict[q_id] = {
             "question": question_text, # Store raw question text
             "question_formatted": question_with_options, # Store formatted version too
             "options": options,
             "answer": answer, # True answer
             "category": category,
             "initial_prompt_used": initial_prompt_data['name'],
             "initial_prompt_file": str(initial_prompt_filepath),
             "critic_prompt_used": critic_prompt_data['name'],
             "critic_prompt_file": str(critic_prompt_filepath),
             "synthesizer_prompt_used": synthesizer_prompt_data['name'],
             "synthesizer_prompt_file": str(synthesizer_prompt_filepath),
             "num_initial_agents": agents, # Store the number of initial agents
             "initial_solutions": initial_solution_messages, # Store initial agent messages
             "critic_response": critic_message, # Store critic's message
             "synthesizer_response": synthesizer_message, # Store synthesizer's message
             # Extract final answer only from the synthesizer's response
             "extracted_answer_synthesizer": parse_answer(synthesizer_message['content']) if synthesizer_message else None
        }


    # --- Save the final results ---
    output_filename_parts = [
        "mmlu_pro_phased", # Indicate phased approach
        f"{agents}initial", # Number of initial agents
        f"{temperature}temp",
        f"{model.replace('/', '_').replace(':', '_')}"
    ]
    output_filename_parts.append(initial_prompt_data['name'])
    output_filename_parts.append(critic_prompt_data['name'])
    output_filename_parts.append(synthesizer_prompt_data['name'])


    output_filename = "_".join(output_filename_parts) + ".json"

    print(f"\nProcessing complete ({len(response_dict)} questions processed).")
    print(f"Saving results to {output_filename}")
    try:
        with open(output_filename, "w") as f:
            json.dump(response_dict, f, indent=4)
        print("Results saved successfully.")
    except IOError as e:
        print(f"Error saving results to file: {e}")


# --- Main script execution and argument parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MMLU-Pro multi-agent (initial -> critic -> synthesis) experiment with configurable prompts and parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments for prompt files
    parser.add_argument(
        "--initial_prompt",
        type=Path,
        required=True,
        help="Filepath for the initial agent prompt YAML."
    )
    parser.add_argument(
        "--critic_prompt",
        type=Path,
        required=True,
        help="Filepath for the critic agent prompt YAML."
    )
    parser.add_argument(
        "--synthesizer_prompt",
        type=Path,
        required=True,
        help="Filepath for the synthesizer agent prompt YAML."
    )


    # Experiment parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of MMLU-Pro questions to process."
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of initial agents to generate solutions."
    )
    # Removed --rounds argument

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the language model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="The language model to use (e.g., 'Qwen/Qwen2.5-7B-Instruct', 'gpt-4-turbo', etc.)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/hgfs/vmshare/NUS_Sem2", # Update default path as needed
        help="Root directory containing the 'dataset/mmlu_pro' folder."
    )

    args = parser.parse_args()

    # --- Check Prompts Directory ---
    # Check if the directory containing prompts exists
    if not args.initial_prompt.parent.exists():
         print(f"Error: Directory for initial prompt '{args.initial_prompt.parent}' not found.")
         exit(1)
    if not args.critic_prompt.parent.exists():
         print(f"Error: Directory for critic prompt '{args.critic_prompt.parent}' not found.")
         exit(1)
    if not args.synthesizer_prompt.parent.exists():
         print(f"Error: Directory for synthesizer prompt '{args.synthesizer_prompt.parent}' not found.")
         exit(1)


    # --- Run Async Main ---
    try:
        asyncio.run(main(
            initial_prompt_filepath=args.initial_prompt,
            critic_prompt_filepath=args.critic_prompt,
            synthesizer_prompt_filepath=args.synthesizer_prompt,
            num_samples=args.num_samples,
            agents=args.agents,
            temperature=args.temperature,
            model=args.model,
            rootdir=args.data_dir,
        ))
    except Exception as e:
        print(f"\nExperiment run failed: {e}")
        # Consider logging the exception traceback in a real application
        # import traceback
        # traceback.print_exc()