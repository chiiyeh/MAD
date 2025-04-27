import json
import openai
import numpy as np
import time
import re
from pathlib import Path
import argparse

def parse_answer(input_str):
    pattern = r'is .*?\((\w)\)'
    # pattern = r'###### (\w)'
    matches = re.findall(pattern, input_str)

    solution = None
    # print("predicted solution")
    # print(input_str)
    # print("matches")
    # print(matches)

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution