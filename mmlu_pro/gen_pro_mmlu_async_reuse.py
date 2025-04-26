import os
import openai
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
from typing import List, Dict, Tuple, Any
from eval_mmlu import parse_answer

# Import libraries for prompt management and argument parsing
import yaml
from jinja2 import Environment, select_autoescape
import argparse

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
DEFAULT_MAX_TOKENS = 2048

#### CHAT TEMPLATE ####

# Modified construct_message function (same logic, just accepts template object)
def construct_message(
    collaborative_template: Any, # Jinja2 Template object
    agents_contexts: List[List[Dict]],
    question_text: str,
    history_index: int
) -> Dict:
    """
    Constructs the user message for the next round using a Jinja2 template,
    incorporating previous responses from other agents.
    """
    other_agents_responses = []
    for agent_context in agents_contexts:
        if len(agent_context) > history_index and agent_context[history_index]["role"] == "assistant":
            agent_response = agent_context[history_index]["content"]
            other_agents_responses.append(agent_response)

    template_vars = {
        "other_agent_responses": other_agents_responses,
        "question": question_text
    }

    user_content = collaborative_template.render(template_vars)

    # Simple fallback check if collaborative content seems insufficient without peers
    if not other_agents_responses and len(user_content.strip()) < 50: # Arbitrary length check
        print("Warning: No other agent responses to include in collaborative prompt. Using fallback.")
        user_content = "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."

    return {"role": "user", "content": user_content}


def construct_assistant_message(completion_text: str) -> Dict:
    """Constructs an assistant message dictionary from completion text."""
    return {"role": "assistant", "content": completion_text}


# --- Async chat function with retry (no changes needed here) ---

@tenacity.retry(
    wait=tenacity.wait_random_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(8),
    retry=tenacity.retry_if_exception_type((
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APIStatusError,
    )),
    before_sleep=lambda retry_state: print(f"Retrying API call (attempt {retry_state.attempt_number}/{retry_state.attempt_number + retry_state.outcome.failures}). Waiting {retry_state.next_action.sleep:.2f}s after error: {retry_state.outcome.exception()}")
)
async def async_chat_with_gpt(
    client: AsyncOpenAI,
    messages: List[Dict],
    model: str, # Model name now comes from arguments
    temperature: float, # Temperature now comes from arguments
    n: int = 1,
    stop: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Tuple[List[str], int]:
    """
    Get completions from GPT model asynchronously with retries.
    """
    try:
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
        raise e


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

def format_example(question, options):
    example = "Question: {}\nOptions:\n".format(question)
    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    return example


# --- Async Main Execution ---
async def main(
    collaborative_prompt_filepath: Path,
    reuse_round: int,
    reuse_file: Path,
    rounds: int,
    temperature: float,
    model: str,
):
    """
    Main asynchronous function to load data, interact with the model,
    and save results, using external prompt templates specified by arguments.
    """
    response_dict: Dict[str, Any] = {}

    # Load prompt templates once at the start using the provided filenames
    print(f"Loading prompt templates: {collaborative_prompt_filepath}")
    try:
        collaborative_prompt_data = load_prompt_template_data(collaborative_prompt_filepath)

        # Get template objects from the environment using the filenames
        collaborative_template = env.from_string(collaborative_prompt_data['template'])

        print(f"Prompt templates loaded successfully: '{collaborative_prompt_data['name']}'.")
    except Exception as e:
        print(f"Failed to load prompt templates. Exiting. Error: {e}")
        # In async main, raising an exception will stop the asyncio run
        raise e

    # Initialize the async OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPINFRA_TOKEN", ""),
        base_url="https://api.deepinfra.com/v1/openai"
    )
    if not client.api_key:
        print("Warning: DEEPINFRA_TOKEN environment variable is not set.")
        print("API calls may fail.")


    # Load and preprocess data (synchronous)
    print("Loading dataset...")
    with open(reuse_file, "r") as f:
        reuse_response_dict = json.load(f)
    first_value = next(iter(reuse_response_dict.values()))
    initial_prompt_used = first_value["initial_prompt_used"]
    initial_prompt_file = first_value["initial_prompt_file"]
    if reuse_round > 1:
        # Will require the collaborative prompt to be the same
        if first_value["collaborative_prompt_file"] != collaborative_prompt_data['name']:
            print(f"Error: Collaborative prompt file {collaborative_prompt_data['name']} does not match the reused data. Exiting.")
            exit(1)
    agents = len(first_value["agent_contexts"])


    output_filename = (
        f"mmlu_pro"
        f"_{agents}agents"
        f"_{rounds}rounds"
        f"_{temperature}temp"
        f"_{model.replace('/', '_')}" # Replace slash in model name for filename safety
        f"_{initial_prompt_used}"
        f"_{collaborative_prompt_data['name']}"
        f".json"
    )

    # Main loop iterating through samples (questions)
    for q_id, question_details in tqdm(reuse_response_dict.items(), desc="Processing questions"):

        # Extract question details
        category = question_details["category"]
        answer = question_details["answer"]
        question_text = question_details["question"]
        options = question_details["options"]
        reuse_agent_contexts = question_details["agent_contexts"]

        print(f"\n--- Question (ID: {q_id}, Subject: {category}) ---")

        # --- Render the initial prompt for all agents ---
        question_with_options = format_example(question_text, options)

        # Initialize agent contexts - from previous rounds
        agent_contexts = [context[:reuse_round*2] for context in reuse_agent_contexts]
        round_num = reuse_round

        # Loop through rounds (synchronous)
        while round_num < rounds:
            print(f" Round {round_num + 1}/{rounds}")

            tasks = [] # List to hold async tasks for API calls in this round
            round_num += 1 # Increment round number

            # Create tasks for each agent's API call in this round
            for agent_idx in range(agents):
                agent_context = agent_contexts[agent_idx]

                # If it's not the first round (round_num > 0), construct the collaborative message
                if round_num > 0:
                    # The index of the assistant message from the *previous* round (round_num - 1)
                    # Assuming messages are added as [user_r0, assistant_r0, user_r1, assistant_r1, ...]
                    previous_assistant_message_index = 2 * (round_num - 1) + 1

                    # Filter to get contexts of *other* agents
                    agent_contexts_other = [agent_contexts[j] for j in range(agents) if j != agent_idx]

                    # Use the collaborative template to create the user message for this round
                    message = construct_message(
                        collaborative_template, # Pass the template object
                        agent_contexts_other,
                        question_with_options, # Pass original question text for templating
                        previous_assistant_message_index
                    )
                    agent_context.append(message) # Add the new user message for this round
                # else: # For round 0, the context was already initialized with the initial prompt above

                # Create the async task for this agent's API call
                task = async_chat_with_gpt(
                    client,
                    agent_context,
                    model=model, # Use model from args
                    temperature=temperature, # Use temperature from args
                    max_tokens=DEFAULT_MAX_TOKENS
                )
                tasks.append(task)

            # Run all agent API calls for this round concurrently
            try:
                results = await asyncio.gather(*tasks)

                # Process the results
                for agent_idx, (completion_list, tokens_used) in enumerate(results):
                    if completion_list:
                        assistant_message = construct_assistant_message(completion_list[0])
                        agent_contexts[agent_idx].append(assistant_message)
                    else:
                        print(f"Warning: No completion received for agent {agent_idx} in round {round_num} after retries.")
                        # Decide how to handle this: maybe add a placeholder error message?
                        # For now, the context for this agent will just not get an assistant response for this round.

            except tenacity.RetryError as e:
                print(f"Error: API call failed after multiple retries in round {round_num} for question ID {q_id}.")
                # The exception from the first failing task is raised by asyncio.gather.
                # agent_contexts for failed agents might not be updated.
            except Exception as e:
                print(f"An unexpected error occurred during concurrent API calls in round {round_num} for question ID {q_id}: {e}")
                # Decide how to handle fatal errors - continue, break, log?
                # For now, print error and continue to next question/round.
                pass

            # Optional: Check for answer convergence across agents here if needed
            # For simplicity, this example runs all planned rounds

        # After all rounds for this question, store the final state
        response_dict[q_id] = {
            "question": question_text,
            "options": options,
            "answer": answer, # True answer
            "category": category,
            "initial_prompt_used": initial_prompt_used, # Track which prompt was used by name
            "initial_prompt_file": initial_prompt_file, # Track which file was used
            "collaborative_prompt_used": collaborative_prompt_data['name'], # Track which prompt was used by name
            "collaborative_prompt_file": str(collaborative_prompt_filepath), # Track which file was used
            "agent_contexts": agent_contexts, # Store the full history for each agent
            "extracted_answers": [parse_answer(ctx[-1]['content']) if ctx and ctx[-1]['role'] == 'assistant' else None for ctx in agent_contexts]
        }

    # Save the final results
    # Include prompt names and other key parameters in the filename for tracking
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
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Reuse rounds from previously ran MMLU-Pro multi-agent experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # Required arguments for prompt files
    parser.add_argument(
        "--collaborative_prompt",
        type=Path,
        required=True,
        help="Filepath for the collaborative prompt YAML for subsequent rounds."
    )
    parser.add_argument(
        "--reuse_round",
        type=int,
        required=True,
        help="Reuse rounds from existing file."
    )
    parser.add_argument(
        "--reuse_file",
        type=Path,
        required=True,
        help="Filepath for the existing results to reuse from."
    )

    # Optional arguments for experiment parameters with default values
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of collaborative rounds for agents to refine answers."
    )
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

    args = parser.parse_args()

    # --- Run Async Main ---
    # Pass the parsed arguments to the main async function
    try:
        asyncio.run(main(
            collaborative_prompt_filepath=args.collaborative_prompt,
            reuse_round=args.reuse_round,
            reuse_file=args.reuse_file,
            rounds=args.rounds,
            temperature=args.temperature,
            model=args.model
        ))
    except Exception as e:
        print(f"\nExperiment run failed: {e}")