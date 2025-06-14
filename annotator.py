import logging
from vllm import LLM, SamplingParams
import re
import os
import json
import torch
import random
from tqdm import tqdm
import yaml
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="annotator.log",
    filemode="w",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

try:
    with open("lang2prompt.yaml", "r", encoding="utf-8") as f:
        lang2prompt = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("File lang2prompt.yaml not found!")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML file: {e}")
    raise

class Annotator:
    def __init__(self, llm: str, quantization: str, temperature: float, batch_size: int):
        logger.info(f"initializing model: {llm}")
        try:
            self.llm = LLM(
                model=llm,
                max_model_len=15_000,
                tensor_parallel_size=1,
                dtype="half",
                gpu_memory_utilization=0.9,
                quantization=quantization,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llm)
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
            
        self.sampling_params = SamplingParams(
            temperature=temperature,
            repetition_penalty=1.1,
            top_k=100,
            max_tokens=1024,
            top_p=0.8,
            stop="<end>"
        )
        self.batch_size = batch_size
        logger.info("model initialized")

    def save_results(self, results, save_path):
        logger.info(f"saving {len(results)} results to {save_path}")
        try:
            with open(save_path, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def create_chat(self, prompt, text):
        return prompt.format(text=text)

    def generate_response(self, chats):
        logger.info(f"generating {len(chats)} responses...")
        try:
            responses = self.llm.generate(chats, self.sampling_params, use_tqdm=False)
            
            logger.info("generation completed")
            return responses
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def get_prompt(self, lang=None):
        if lang is None:
            logger.error("Language not specified for prompt retrieval.")
            raise ValueError("Language must be specified to retrieve prompt.")

        if lang not in lang2prompt:
            logger.error(f"No prompt found for language: {lang}")
            raise KeyError(f"No prompt found for language: {lang}")
        
        prompt_text = lang2prompt[lang]
        
        messages = [
            {
                "role": "system",
                "content": "You are accurate and precise entity recognition model that extracts named entities from arbitrary text.",
            },
            {"role": "user", "content": prompt_text},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

        return prompt


    def annotate_this_dataset(self, dataset, process_response=None, save_steps=100, save_path=None):
        results = {}
        counter = 0
        buffer = []
        logger.info("starting annotation loop")
        
        lang_groups = {}
        for example in dataset:
            if 'lang' not in example or 'text' not in example:
                logger.warning(f"Skipping example with missing 'lang' or 'text' keys: {example}")
                continue
            
            lang = example['lang']
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append(example)
        total_examples = sum(len(exs) for exs in lang_groups.values())
        with tqdm(total=total_examples, desc="Total Progress", unit="example") as overall_pbar:
            for lang, examples in lang_groups.items():
                logger.info(f"Processing {len(examples)} examples for language: {lang}")
                
                try:
                    prompt = self.get_prompt(lang)
                except KeyError:
                    overall_pbar.update(len(examples))
                    logger.error(f"Skipping language {lang} - no prompt available")
                    continue
                
                for i in range(0, len(examples), self.batch_size):
                    batch = examples[i:i + self.batch_size]
                    texts = [example['text'] for example in batch]
                    
                    chats = [self.create_chat(prompt, text) for text in texts]
                    
                    try:
                        responses = self.generate_response(chats)
                        for i, resp in enumerate(responses):
                            logger.info(f"RAW[{lang}]: {resp.outputs[0].text}")
                        if process_response:
                            batch_results = process_response(responses, texts)
                            buffer.extend(batch_results)
                        else:
                            for j, response in enumerate(responses):
                                if j < len(texts):
                                    result = {
                                        'text': texts[j],
                                        'response': response.outputs[0].text,
                                        'lang': lang
                                    }
                                    buffer.append(result)
                        
                        counter += len(batch)
                        overall_pbar.update(len(batch))
                        logger.info(f"Processed {counter} examples total")
                        
                        if len(buffer) >= save_steps and save_path:
                            self.save_results(buffer, save_path)
                            buffer.clear()
                            logger.info(f"Saved intermediate results at step {counter}")
                            
                    except Exception as e:
                        logger.error(f"Error processing batch for language {lang}: {e}")
                        overall_pbar.update(len(batch))
                        continue
        
        if buffer and save_path:
            self.save_results(buffer, save_path)
            logger.info(f"Saved final batch of {len(buffer)} results")
        
        results['total_processed'] = counter
        results['languages_processed'] = list(lang_groups.keys())
        
        logger.info(f"Annotation completed. Total processed: {counter}")
        return results

    def process_ner_responses(self, responses, texts):
        pattern = re.compile(r'([^<>\n]+)\s*<>\s*([^<>\n]+)', re.MULTILINE)
        batch_results = []
        
        for i, response in enumerate(responses):
            if i >= len(texts):
                logger.warning(f"Response index {i} exceeds texts length {len(texts)}")
                continue
                
            text = texts[i]
            result = response.outputs[0].text
            
            matches = pattern.findall(result)
            
            if matches:
                items = [f"{entity.strip()}<>{label.strip()}" for entity, label in matches]
                batch_results.append({'text': text, 'labels': items})
                logger.info(f"Extracted {len(items)} entities from text: {text[:50]}...")
            else:
                logger.warning(f"No pattern match found for text: {text[:50]}...")
                batch_results.append({'text': text, 'labels': []})
                
        return batch_results

    def run(self, dataset, save_path):
        logger.info("starting run() method")
        processor = self.process_ner_responses
        results = self.annotate_this_dataset(
            dataset=dataset,
            process_response=processor,
            save_steps=1,
            save_path=save_path
        )
        logger.info("run() completed")
        return results
