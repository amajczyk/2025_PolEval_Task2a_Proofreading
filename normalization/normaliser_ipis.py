import sys
import json
import logging
from tqdm.auto import tqdm
import torch

from pathlib import Path
from typing import List, Dict
import re

from combo.predict import COMBO
from lambo.segmenter.lambo import Lambo

aster = re.compile('.+\*.+')
back = re.compile('.+/.+')

lambo = Lambo.get("Polish")

# UPDATE 2024-10-24: Due to temporary server issues, automatic combo-model downloading may be unavailable.
#                    If you are unable to download the model directly via ```combo = COMBO.from_pretrained('polish-herbert-base-ud213')```,
#                    please download the model manually from:
#                    https://mozart.ipipan.waw.pl/~alina/Polish_dependency_parsing_models/COMBO3/polish-herbert-base-ud213.tar.gz
#                    Add a path to the model: ```combo = COMBO.from_pretrained('path-to-the-combo-model/polish-herbert-base-ud213')```

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# combo = COMBO.from_pretrained('/mnt/d/Pobrane/poleval-gender/combo_model/polish-herbert-base-ud213.tar.gz', cuda_device=device)
combo = COMBO.from_pretrained('/media/adam/Samsung/Documents/PAN/for paper - all models/polish-herbert-base-ud213.tar.gz', cuda_device=device)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UPDATE 2024-10-24: Run the normalisation script from the command line as follows:
#                    ```python normaliser_ipis.py sample_proofreading.jsonl sample_proofreading_normalised.jsonl```


def read_instances(path: Path) -> List[Dict]:
    try:
        with open(path, 'r') as f:
            # Iterate over each line in the file, parse as JSON, and append to instances list
            instances = [json.loads(line) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {path} was not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"An error decoding JSON in file {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading {path}: {e}")

    return instances


def write_instances(path: Path, instances: List[Dict]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            # Serialize the list of instances into JSON, one per line
            for instance in instances:
                json.dump(instance, f, ensure_ascii=False)
                f.write("\n")
    except IOError as e:
        raise RuntimeError(f"An error occurred while writing to {path}: {e}")


def normalisation_of_gendered_texts(text: str) -> Dict[int, str]:
    """
    Normalise gendered forms in the given text based on predefined patterns.
    Note that the script only handles forms with a gender star and masculatives and feminativs joined with a conjunction or /
    """
    final = {}
    for i, ln in enumerate(text.strip().split('\n')):
        normalized_words = [normalize_form_word(word) for word in ln.split()]
        final[i] = ' '.join(normalized_words)
    return final


def normalize_form_word(word: str) -> str:
    """Normalise individual word based on its structure."""
    if aster.search(word):
        # Handle double forms with a gender star, e.g. psycholo*giem/żką
        parts = word.split("*")
        if back.search(parts[1]):
            # Split masculine/feminine endings
            endings = parts[1].split('/')
            return f"{parts[0]}{endings[0]} {parts[0]}{endings[1]}"
        else:
            return f"{parts[0]} {parts[0]}{parts[1]}"
    elif back.search(word):
        # Handle masculine and feminine full forms joined with '/', e.g. psychologiem/psycholożką
        return ' '.join(word.split("/"))
    else:
        return word


def nlp(comb: COMBO, lamb: Lambo, text: str) -> list[list]:
    """
    Sentence segmenation, tokenisation and filtering of punctuation marks (PUNCT) and conjunctions (CCONJ, SCONJ)
    Note that the text passages are relativelly short and can be processed by COMBO using HerBERT.
    If an LLM generates a passage exceeding 500 tokens, it is likely an error.
    """
    
    filtered_tokens = []
    # logger.info("Starting normalization of input text.")
    normalised_texts = normalisation_of_gendered_texts(text)
    # logger.info(f"Normalised into {len(normalised_texts)} text passages.")

    for i in range(len(normalised_texts)):
        tokenised_text = lamb.segment(normalised_texts[i])

        for turn in tokenised_text.turns:
            for sentence in turn.sentences:
                if len(sentence.tokens) > 500:
                    logger.warning(f"Sentence >>{sentence.text}<< too long (>500 tokens). Shortening.")
                    predictions = comb(' '.join([token.text for token in sentence.tokens[:290]]))
                else:
                    predictions = comb(sentence.text)

                for predicted_sentence in predictions:
                    for token in predicted_sentence.tokens:
                        if token.upostag not in {"PUNCT", "CCONJ", "SCONJ"}:
                            filtered_tokens.append(token.text.lower())
    return filtered_tokens


def normaliser(comb: COMBO, lamb: Lambo, input_path: Path, output_path: Path) -> None:
    # Read instances from input
    json_instances = read_instances(input_path)

    # Normalise each instance
    normalised_instances = [
        {**instance, 'normalised_target': nlp(comb, lamb, instance['target'])}
        for instance in tqdm(json_instances, total=len(json_instances), desc="Normalising instances", unit="instance")
    ]
    logger.info(f"Processed {len(json_instances)} instances.")

    # Write normalized instances to the output path
    write_instances(output_path, normalised_instances)
    logger.info(f"Normalized instances saved to {output_path}")


def main(comb: COMBO, lamb: Lambo, in_jsonl: str, out_jsonl: str) -> None:
    normaliser(comb, lamb, Path(in_jsonl), Path(out_jsonl))


if __name__ == '__main__':

    input_jsonl = sys.argv[1]
    normalised_jsonl = sys.argv[2]
    
    main(combo, lambo, input_jsonl, normalised_jsonl)
