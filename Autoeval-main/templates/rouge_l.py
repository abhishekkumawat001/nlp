import math
from typing import List
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import IPython.display as display


def rouge_l(prediction: str, reference: str, beta=1) -> float:
    
    # BEGIN CODE : eval1.rouge_l
    # >>> {segment:eval1.rouge_l} <<<

    # END CODE
    pass
    
