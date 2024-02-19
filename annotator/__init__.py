import os
import json
import uuid
import time
import random
import openai
import requests
from tqdm import tqdm
from typing import List, Optional, Union
from .instructions import create_prompt, reject_response



class GeneralGenerator:
    """
    base class for systematically augment dataset
    """

    def __init__(self, model_name, backend="openai", backend_settings={}) -> None:
        self.model_name = model_name
        if backend == "openai":
            from .autocompletes.openai import OpenAICompletion

            self.backend = OpenAICompletion(**backend_settings)
        elif backend == "anthropic":
            from .autocompletes.anthropic import ClaudeCompletion

            self.backend = ClaudeCompletion(**backend_settings)
        elif backend == "test":
            self.backend = None
        else:
            raise ValueError(
                f"backend {backend} is currently not supported, supported backends openai, anthropic"
            )

    def sample_task(self, available_tasks: list):
        if isinstance(available_tasks[0], str):
            task_choice = available_tasks
            task_weights = [1 / len(task_choice) for _ in range(len(available_tasks))]
        elif isinstance(available_tasks[0], tuple):
            task_choice = [t[0] for t in available_tasks]
            task_weights = [t[1] for t in available_tasks]
        c = random.choices(task_choice, task_weights, k=1)[0]
        return c

    def augment(self, text: Union[str, list], available_tasks: list, model_name):
        task = self.sample_task(available_tasks)
        input_text = create_prompt(text, task)
        
    
        text, response = self.backend.completion(input_text, model_name=self.model_name, temperature=1, top_p=0.9)
        # text, response = self.backend.completion(input_text, model_name=self.model_name, temperature=1, top_p=0.9)
      

        return input_text, response, task

    def get_strategy(self, strategy):
            return [strategy,]

    def cache_result(self, new_prompt, res, task, input):
        with open(self.save_file_path, "a") as fout:
            fout.write(
                json.dumps(
                    {
                        "instruction": input,
                        "output": res,

                    }
                )
                + "\n"
            )

    def ingest(
        self, texts, save_file_path, strategy="all", model_name="gpt-4-1106-preview"
    ):
        added = set()
       
        if os.path.exists(save_file_path):
            with open(save_file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line[0] == '{':
                        payload = eval(line)
                        added.add(payload["instruction"])
                        added.add(payload["output"])

        self.save_file_path = save_file_path
        available_task = self.get_strategy(strategy)

        for text in tqdm(texts, dynamic_ncols=True):
            if text not in added:
                # wait in order not to touch the openai rate limit
                time.sleep(10)
                new_prompt, res, task = self.augment(text, available_task, model_name)

                if reject_response(new_prompt):
                    # we will try it in the next round
                    added.add(text)
                    continue
                self.cache_result(new_prompt, res, task, text)
                added.add(new_prompt)
       
        print("augmentation finished!")
        augmented = []
        with open(self.save_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '{':
                    payload = eval(line)
                    augmented.append(payload["output"])

        return augmented
