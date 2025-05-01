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
from dataclasses import dataclass

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
DEFAULT_MAX_TOKENS = 3000

@dataclass
class DebateTurn:
    speaker: str
    argument: str

@dataclass
class DebateHistory:
    turns: List[DebateTurn]

    def add_turn(self, speaker: str, argument: str):
        self.turns.append(DebateTurn(speaker, argument))

    def get_transcript(self) -> str:
        return "\n".join([f"{t.speaker}: {t.argument}" for t in self.turns])


class JudgeAgent:
    def __init__(self, llm, max_rounds=6):
        self.llm = llm
        self.max_rounds = max_rounds

    def check_consensus(self, history: DebateHistory) -> bool:
        prompt = f"Transcript:\n{history.get_transcript()}\n\nHave the debaters reached a consensus? Answer 'Yes' or 'No'."
        response = self.llm(prompt).strip().lower()
        return "yes" in response

    def summarize_and_decide(self, history: DebateHistory) -> str:
        prompt = f"""
            Transcript of a debate:
            {history.get_transcript()}

            Summarize the main points from both sides. Then decide who made a stronger case: Affirmative or Negative? Justify your answer.
            """
        return self.llm(prompt).strip()


#### CHAT TEMPLATE ####

# Modified construct_message function (same logic, just accepts template object)
def construct_message(
    agent_template: Any, # Jinja2 Template object
    debate_history: DebateHistory,
) -> Dict:
    """
    Constructs the user message for the next round using a Jinja2 template,
    incorporating previous responses from other agents.
    """
    debate_chain = []
    for debate_turn in debate_history.turns:
        agent_response = debate_turn.argument["content"]
        debate_chain.append(agent_response)

    template_vars = {
        "debate_chain": debate_chain
    }

    user_content = agent_template.render(template_vars)

    # Simple fallback check if collaborative content seems insufficient without peers
    if not agent_response and len(user_content.strip()) < 50: # Arbitrary length check
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

semaphore = asyncio.Semaphore(10)  # Adjust the concurrency level based on your API rate limits
response_lock = asyncio.Lock()


# --- Async Main Execution ---
async def main(
    initial_judge_prompt_filepath: Path,
    initial_debater_prompt_filepath: Path,
    debater_1_prompt_filepath: Path,
    debater_2_prompt_filepath: Path,
    num_samples: int,
    agents: int,
    rounds: int,
    temperature: float,
    model: str,
    rootdir: str, # Pass rootdir as arg
    resume: bool,
):
    """
    Main asynchronous function to load data, interact with the model,
    and save results, using external prompt templates specified by arguments.
    """
    response_dict: Dict[str, Any] = {}

    # Load prompt templates once at the start using the provided filenames
    print(f"Loading prompt templates: {initial_judge_prompt_filepath}, {initial_debater_prompt_filepath}, {debater_1_prompt_filepath}, {debater_1_prompt_filepath}")
    try:
        initial_judge_prompt_data = load_prompt_template_data(initial_judge_prompt_filepath)
        initial_debater_prompt_data = load_prompt_template_data(initial_debater_prompt_filepath)
        debater_1_prompt_data = load_prompt_template_data(debater_1_prompt_filepath)
        debater_2_prompt_data = load_prompt_template_data(debater_2_prompt_filepath)

        # Get template objects from the environment using the filenames
        initial_judge_template = env.from_string(initial_judge_prompt_data['template'])
        initial_debater_template = env.from_string(initial_debater_prompt_data['template'])
        debater_1_template = env.from_string(debater_1_prompt_data['template'])
        debater_2_template = env.from_string(debater_2_prompt_data['template'])

        print(f"Prompt templates loaded successfully: '{initial_judge_prompt_data['name']}', {initial_debater_prompt_data['name']}, '{debater_1_prompt_data['name']}' and '{debater_2_prompt_data['name']}'.")
    except Exception as e:
        print(f"Failed to load prompt templates. Exiting. Error: {e}")
        # In async main, raising an exception will stop the asyncio run
        raise e

    # Initialize the async OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPINFRA_TOKEN", "0S0QjhkqiBcFJP8Gr47e1WfDe17YZDVJ"),
        base_url="https://api.deepinfra.com/v1/openai"
    )
    if not client.api_key:
        print("Warning: DEEPINFRA_TOKEN environment variable is not set.")
        print("API calls may fail.")


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


    output_filename = (
        f"mmlu_pro"
        f"_{agents}agents"
        f"_{rounds}rounds"
        f"_{temperature}temp"
        f"_{model.replace('/', '_')}" # Replace slash in model name for filename safety
        f"_{initial_judge_prompt_data['name']}"
        f"_{debater_1_prompt_data['name']}"
        f".json"
    )

    if resume and os.path.exists(output_filename):
        print(f"Resuming from existing results in {output_filename}.")
        try:
            with open(output_filename, "r") as f:
                response_dict = json.load(f)
            print(f"Loaded {len(response_dict)} previously processed questions.")
        except IOError as e:
            print(f"Error loading existing results: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from existing results: {e}")
            return
        except Exception as e:
            print(f"Unexpected error loading existing results: {e}")
            return

    async def process_question(i):
        async with semaphore:
            subject = random.choice(subjects)
            df = test_df[subject]

            if not df or len(df) == 0:
                print(f"Warning: No questions found for subject {subject}. Skipping.")
                return

            idx = random.randint(0, len(df) - 1)
            single_question = df[idx]
            q_id = single_question.get("question_id", f"{subject}_{idx}")
            answer = single_question["answer"]
            category = single_question["category"]
            question_text = single_question["question"]
            options = single_question["options"]

            async with response_lock:
                if q_id in response_dict:
                    print(f"Question ID {q_id} already processed. Skipping.")
                    return

            print(f"\n--- Question {i + 1}/{num_samples} (ID: {q_id}, Subject: {category}) ---")

            question_with_options = format_example(question_text, options)
            initial_prompt_vars = {
                "category": category,
                "question": question_with_options,
            }

            # Render templates
            initial_judge_prompt_content = initial_judge_template.render(initial_prompt_vars)
            debater_1_prompt_content = initial_debater_template.render(initial_prompt_vars)
            debater_2_prompt_content = initial_debater_template.render(initial_prompt_vars)

            agent_contexts = [
                [{"role": "user", "content": debater_1_prompt_content if i % 2 == 0 else debater_2_prompt_content}] for
                i in range(agents - 1)]
            agent_contexts.append(([{"role": "user", "content": initial_judge_prompt_content}]))

            debate_history = DebateHistory(turns=[])

            for round_num in range(rounds):
                print(f" Round {round_num + 1}/{rounds}")
                tasks = []

                for agent_idx in range(agents):
                    agent_context = agent_contexts[agent_idx]

                    if agent_idx == agents - 1:
                        message = construct_message(agent_template=initial_judge_template,
                                                    debate_history=debate_history)
                        agent_context.append(message)
                    elif round_num > 0:
                        message = construct_message(
                            agent_template=debater_1_template if agent_idx % 2 == 0 else debater_2_template,
                            debate_history=debate_history,
                        )
                        agent_context.append(message)

                    try:
                        completion_list, tokens_used = await async_chat_with_gpt(
                            client,
                            agent_context,
                            model=model,
                            temperature=temperature,
                            max_tokens=DEFAULT_MAX_TOKENS,
                        )

                        if completion_list:
                            assistant_message = construct_assistant_message(completion_list[0])
                            agent_contexts[agent_idx].append(assistant_message)
                            if agent_idx != agents - 1:
                                debate_history.add_turn(agent_idx, assistant_message)
                        else:
                            print(f"Warning: No completion for agent {agent_idx} in round {round_num}.")
                    except tenacity.RetryError:
                        print(f"RetryError in round {round_num} for question ID {q_id}.")
                    except Exception as e:
                        print(f"Unexpected error: {e}")

            # Store final state
            response_data = {
                "question": question_text,
                "options": options,
                "answer": answer,
                "category": category,
                "initial_prompt_used": initial_judge_prompt_data['name'],
                "initial_prompt_file": str(initial_judge_prompt_filepath),
                "collaborative_prompt_used": debater_1_prompt_data['name'],
                "collaborative_prompt_file": str(debater_1_prompt_filepath),
                "agent_contexts": agent_contexts,
                "extracted_answers": [
                    parse_answer(ctx[-1]['content']) if ctx and ctx[-1]['role'] == 'assistant' else None
                    for ctx in agent_contexts
                ]
            }

            async with response_lock:
                response_dict[q_id] = response_data

    # Main loop iterating through samples (questions)
    tasks = [process_question(i) for i in range(num_samples)]
    for coro in tqdm(asyncio.as_completed(tasks), total=num_samples, desc="Processing questions"):
        await coro

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
        description="Run MMLU-Pro multi-agent experiment with configurable prompts and parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # Required arguments for prompt files
    parser.add_argument(
        "--initial_judge_prompt",
        type=Path,
        required=True,
        help="Filepath for the initial agent prompt YAML."
    )
    parser.add_argument(
        "--debater_1_prompt",
        type=Path,
        required=True,
        help="Filepath for the prompt YAML for debater persona 1."
    )

    parser.add_argument(
        "--debater_2_prompt",
        type=Path,
        required=True,
        help="Filepath for the prompt YAML for debater persona 2."
    )

    # Optional arguments for experiment parameters with default values
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
        help="Number of independent agents to simulate."
    )
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
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/hgfs/vmshare/NUS_Sem2", # Update default path as needed
        help="Root directory containing the 'dataset/mmlu_pro' folder."
    )

    # Additional optional arguments for resuming from existing results
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results if available."
    )

    parser.add_argument(
        "--initial_debater_prompt",
        type=Path,
        required=True,
        help="Filepath for the prompt YAML for initial debater."
    )


    args = parser.parse_args()

    # --- Run Async Main ---
    # Pass the parsed arguments to the main async function
    try:
        asyncio.run(main(
            initial_judge_prompt_filepath=args.initial_judge_prompt,
            initial_debater_prompt_filepath=args.initial_debater_prompt,
            debater_1_prompt_filepath=args.debater_1_prompt,
            debater_2_prompt_filepath=args.debater_2_prompt,
            num_samples=args.num_samples,
            agents=args.agents,
            rounds=args.rounds,
            temperature=args.temperature,
            model=args.model,
            rootdir=args.data_dir,
            resume=args.resume,
        ))
    except Exception as e:
        print(f"\nExperiment run failed: {e}")