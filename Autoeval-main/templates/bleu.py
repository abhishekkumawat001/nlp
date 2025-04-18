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


def bleu_score(candidate: str, reference: str) -> float:
    # BEGIN CODE : eval2.bleu_score
    # >>> {segment:eval2.bleu_score} <<<
    
    # END CODE
    pass
    