from torch.utils.data.dataset import Dataset
import json
import os
import pickle
import random
import time
import warnings
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import torch
import re 

# import logger as logging

import logging
# from torch.utils.data.dataset import Dataset

from filelock import FileLock

logger = logging.getLogger(__name__)


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    

    def __init__(
        self,
        tokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
        device = 'cpu'
    ):
        self.device = device
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                text_list = list()
                with open(file_path, encoding="utf-8") as f:
                    for line in f.readlines():
                        text_list.append(line)
                
                tokens_list = list()
                for text in tqdm(text_list,position=0,leave=False,desc=f'{file_path}_tokenize'):
                    # \n  줄바꿈 문자 추가 
                    # if "\n" in text: 
                    text = re.sub(r'\n',"",text)
                    tokens_list.append(tokenizer.tokenize(text))

                tokenized_text = list()

                for tokens in tqdm(tokens_list,position=0,leave=False,desc=f'{file_path}_tokens_to_ids'):
                    tokenized_text.extend(tokenizer.convert_tokens_to_ids(tokens))


                # tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long).to(self.device)