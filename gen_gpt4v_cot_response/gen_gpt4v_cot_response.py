import sys
sys.path.append('.')
import os
import re
import time
import requests
import tiktoken
import logging
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from utils.dataset_utils.data_config import data_configs
from utils.tools import load_csv, save_csv, encode_image
from prompts import PROMPT, PARSE_PATTERN

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_exception


enc = tiktoken.get_encoding("cl100k_base")
load_dotenv()
api_key = os.getenv("API_KEY")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def get_response(
    prompt: str, 
    image_path: str,
    model_name: str, 
    regex_pattern: Optional[str] = None, 
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seeds: List[int] = [42, 43, 44, 45, 46],
    service_url: str = 'https://api.openai.com/v1/chat/completions',
    seed_idx: int = 0,
    service_error_try: int = 0, 
    parse_error_try: int = 0,
) -> Tuple[str]:
    """
    Get the response from the OpenAI API based on the given prompt.

    Args:
        prompt (str): The user's prompt.
        model_name (str): The name of the model to use.
        image_path (str): The path to the image.
        regex_pattern (Optional[str], optional): The regex pattern to match in the response. Defaults to None.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 1.0.
        top_p (float, optional): The top-p parameter for generating the response. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 1024.
        seeds (List[int], optional): The list of seed values. Defaults to [42, 43, 44, 45, 46].
        seed_idx (int, optional): The index of the seed value to use. Defaults to 0.
        service_url (str, optional): URL of the OpenAI API service. Defaults to 'https://api.openai.com/v1/chat/completions'.
        service_error_try (int, optional): The number of times to retry in case of service errors. Defaults to 0.
        parse_error_try (int, optional): The number of times to retry in case of parsing errors. Defaults to 0.

    Returns:
        Tuple[str]: A tuple containing the response from the OpenAI API.
    """

    if len(enc.encode(prompt)) > 3800:
        prompt = enc.decode(enc.encode(prompt)[:3800])
    base64_image = encode_image(image_path)
    payload = {
        "model": model_name,
        "messages": [{
            'role': 'user',
            'content': [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ]
        }],
        "seed": seeds[seed_idx],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        response = ''
        response = requests.post(service_url, headers=headers, json=payload).json()
        if 'system_fingerprint' in response:
            logging.info(f'Model {model_name} request finish, system fingerprint: {response["system_fingerprint"]}')
        else:
            logging.info(f'Model {model_name} request finish, no system fingerprint in response')
        response = response['choices'][0]['message']['content'].strip()
        if len(re.findall(regex_pattern, response)) == 0:
            raise ValueError("parse pattern not detected.")
        elif len(re.findall(regex_pattern, response)) > 1:
            raise ValueError("Detect more than on matched result.")
        if type(re.findall(regex_pattern, response)[0]) == tuple:
            return re.findall(regex_pattern, response)[0]
        else:
            return (re.findall(regex_pattern, response)[0], )
    except (KeyError, requests.exceptions.ProxyError, requests.exceptions.SSLError, requests.exceptions.ConnectionError, ConnectionRefusedError) as e:
        # Service Error
        logging.error(f'Get Service Error {e.__class__.__name__}: {e}')
        logging.error(f'The trigger text is {response}')
        time.sleep(2**service_error_try)
        return get_response(prompt, image_path, model_name, regex_pattern, temperature, top_p, max_tokens, seeds, service_url, seed_idx, service_error_try, parse_error_try+1)
    except (ValueError, IndexError) as e:
        # Parsing Error
        logging.error(f'Get Parsing Error {e.__class__.__name__}: {e}')
        logging.error(f'The trigger text is {response}')
        if parse_error_try > 4:
            return (None, )
        time.sleep(2**service_error_try)
        return get_response(prompt, image_path, model_name, regex_pattern, temperature, top_p, max_tokens, seeds, service_url, seed_idx, service_error_try, parse_error_try+1)

def main():
    for dataset_name, dataset_config in data_configs.items():
        # obtain all gpt4v responses
        gpt4v_responses = {}
        for target_name in dataset_config.target_names:
            all_datas = load_csv(f'{dataset_config.data_dir}/in-target/{dataset_config.short_target_names[target_name]}/train.csv')
            all_datas += load_csv(f'{dataset_config.data_dir}/in-target/{dataset_config.short_target_names[target_name]}/valid.csv')
            all_datas += load_csv(f'{dataset_config.data_dir}/in-target/{dataset_config.short_target_names[target_name]}/test.csv')
            for data in all_datas:
                image_path = data['tweet_image']
                prompt = PROMPT % (data['tweet_text'], data['stance_target'])
                response = get_response(prompt, image_path, 'gpt-4o-2024-05-13', regex_pattern=PARSE_PATTERN)
                gpt4v_responses[f"{data['tweet_image']}:{data['stance_target']}:{data['tweet_image']}"] = response[0]

        # save gpt4v responses
        # in-target
        for target_name in dataset_config.target_names:
            for data_split in ['train', 'valid', 'test']:
                all_datas = load_csv(f'{dataset_config.data_dir}/in-target/{dataset_config.short_target_names[target_name]}/{data_split}.csv')
                for data in all_datas:
                    data['gpt4v_response'] = gpt4v_responses[f"{data['tweet_image']}:{data['stance_target']}:{data['tweet_image']}"]
                if not os.path.exists(f'{dataset_config.data_dir}/in-target/{dataset_config.short_target_names[target_name]}'):
                    os.makedirs(f'{dataset_config.data_dir}/in-target/{dataset_config.short_target_names[target_name]}')
                save_csv(f'{dataset_config.data_dir}/in-target/{dataset_config.short_target_names[target_name]}/{data_split}.csv', all_datas)

        # zero-shot
        if dataset_name == 'mccq':
            continue
        for target_name in dataset_config.zero_shot_target_names:
            for data_split in ['train', 'valid', 'test']:
                all_datas = load_csv(f'{dataset_config.data_dir}/zero-shot/{dataset_config.short_target_names[target_name]}/{data_split}.csv')
                for data in all_datas:
                    data['gpt4v_response'] = gpt4v_responses[f"{data['tweet_image']}:{data['stance_target']}:{data['tweet_image']}"]
                if not os.path.exists(f'{dataset_config.data_dir}/zero-shot/{dataset_config.short_target_names[target_name]}'):
                    os.makedirs(f'{dataset_config.data_dir}/zero-shot/{dataset_config.short_target_names[target_name]}')
                save_csv(f'{dataset_config.data_dir}/zero-shot/{dataset_config.short_target_names[target_name]}/{data_split}.csv', all_datas)


if __name__ == '__main__':
    main()