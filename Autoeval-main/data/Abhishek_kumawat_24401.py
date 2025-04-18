# %% [markdown]
# # Assignment 4: Evaluation Metrics + Manipulating Attention Heads in Transformers (TA: Karan Raj Bagri)
# 
# The goal of this assignment is to show how different attention heads have varying significance for transformers trained on multiple tasks by computing gradient-based importance scores, selectively pruning less critical heads, and evaluating the impact on performance.
# 
# Please make a copy of this notebook (locally or on Colab). Ensure you adhere to the guidelines and submission instructions (mentioned below) for attempting and submitting the assignment.
# 
# Given that the class has 150+ students, we will **NOT** entertain any requests for changing your notebooks after the submission deadline (especially in cases when the notebook fails to compile or run because you did not follow the instructions).

# %% [markdown]
# ### Guidelines for Attempting the Assignment
# 
# 1. Write your logic in the cells **ONLY** which have the comment `# ADD YOUR CODE HERE`, between the `# BEGIN CODE` and `# END CODE` comments. These cells are also demarcated by the special start (`## ==== BEGIN EVALUATION PORTION`) and end (`## ==== END EVALUATION PORTION`) comments. Do **NOT** remove any of these comments from the designated cells, otherwise your assignment will not be evaluated correctly.
# 
# 2. All imports that should be necessary are already provided as part of the notebook. You should **NOT** import any new libraries, otherwise your assignment will not be graded.
# 
# 3. You need to install the libraries/imports used in this notebook yourself. Its recommended to use python version between 3.9 and 3.11 to attempt this assignment.
# 
# 4. Only write your code in the cells designated for auto-evaluation. If you encounter any errors in the supporting cells during execution, contact the respective TAs.
# 
# 5. **Please read the function docs and comments carefully**. They provide specific instructions and examples for implementing each function. Follow these instructions precisely - neither oversimplify nor overcomplicate your implementations. Deviating from the provided implementation guidelines may result in lost marks.
# 
# 6. **Important**: Use of AI-assistive technologies such as ChatGPT or GitHub CoPilot is not permitted for this assignment. Ensure that all attempts are solely your own. Not following this rule can incur a large penalty, including but not limited to scoring a zero for this assignment.

# %% [markdown]
# ### Submission Instructions
# 
# 1. Ensure your code follows all guidelines mentioned above before submission.
# 
# 2. Ensure you only add code in designated areas, otherwise your assignment will not evaluated.
# 
# 3. When you have completely attempted the assignment, export the current notebook as a `.py` file, with the following name: `SAPName_SRNo_assignment4.py`, where `SAPName` would be your name as per SAP record, and `SRNo` will be the last 5 digits of your IISc SR number. For example, IISc student with SAP name Twyla Linda (SR no - 04-03-00-10-22-20-1-15329) would use `Twyla_Linda_15329_assignment4.py`.
# 
# 4. You should put your assignment file `SAPName_SRNo_assignment4.py` inside a folder `SAPName_SRNo`. The folder structure looks as follows:
# 
# ``` python
# └─── SAPName_SRNo
#      ├─── SAPName_SRNo_assignment4.py
# ```
# 
# 5. When you run the assignment code, it may download certain datasets and other artifacts. These should **NOT** be part of the above folder.
# 
# 6. Once you have validated the folder structure as above, zip the folder and name it as `SAPName_SRNo_assignment4.zip` and submit this ZIP archive.
# 
# 7. Please ensure that your notebook finishes execution of all the cells within 30 minutes.

# %% [markdown]
# ### Marks Distribution
# 
# - Evaluation Metrics: 30 marks
# - Head Importance Analysis: 20 marks
# - Pruning: 50 marks

# %% [markdown]
# # Imports

# %%
# !pip install datasets

# %%
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

# %% [markdown]
# You can implement the `compute_head_importance` method using any one of the following approaches—or even propose your own method.
# 
# 1. **Gradient-based scoring**:  
#    Compute the gradients of the loss with respect to the head masks (as in [Michel et al. (2019)](https://arxiv.org/abs/1905.10650)). Larger gradient magnitudes indicate that a head has a larger impact on the loss, and hence is more important.
# 
# 2. **Simple ablation**:  
#    Systematically mask (disable) each attention head one at a time and measure the performance drop (e.g., in terms of loss). Heads whose removal causes larger performance degradation are deemed more important.
# 
# 3. **Attention weight magnitude**:  
#    Average the attention weights for each head across samples and time steps. Heads with higher average attention weights might be considered more influential.
# 
# 4. **Confidence analysis**:  
#    Evaluate the sharpness or concentration of each head's attention distribution. Heads that produce less uniform (more confident) distributions may be more specialized and important.
# 
# 5. **Layer-wise relevance propagation (LRP)**:  
#    Use relevance propagation techniques (as in [Voita et al. (2019)](https://arxiv.org/abs/1905.09418)) to trace how much each head contributes to the final output.
# 

# %% [markdown]
# ## Evaluation Metrics

# %% [markdown]
# ### ROUGE-L
# 
# The ROUGE-L metric evaluates text summarization quality by measuring the Longest Common Subsequence (LCS) between a machine-generated summary and a reference summary.
# 
# ROUGE-L is particularly useful for assessing how well a summary captures the main ideas while maintaining sentence-level structure similarity. It doesn't require consecutive matches, making it more flexible than strict n-gram methods.
# 
# You may look at section 3.1 of the [original ROUGE paper by Lin (2004)](https://aclanthology.org/W04-1013/) and implement sentence-level LCS.

# %%
## ==== BEGIN EVALUATION PORTION
def rouge_l(prediction: str, reference: str, beta=1) -> float:
    """
    Computes the ROUGE-L F measure based on the Longest Common Subsequence (LCS).

    ROUGE-L measures the similarity between summaries by finding the longest sequence
    of words that appear in the same relative order in both texts. It computes precision
    (the proportion of words in the prediction that are part of the LCS) and recall
    (the proportion of words in the reference that are part of the LCS), and then combines
    them using an F measure.

    The F measure is computed as:
        F_beta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
    where beta determines the weight of recall relative to precision.
    - A beta of 1.0 (default) gives equal weight to precision and recall (the F1 score).
    - A beta greater than 1 emphasizes recall, while a beta less than 1 emphasizes precision.

    Parameters:
        prediction (str): The machine-generated or candidate summary text.
        reference (str): The human-written or ground truth summary text.
        beta (float): The weight factor for recall relative to precision. Default is 1.0.

    Returns:
        float: The ROUGE-L F measure between 0.0 and 1.0, where higher values indicate
               greater similarity between the prediction and reference.

    Example:
        >>> rouge_l("The cat is on the mat", "The cat sat on the mat")
        0.83  # (approximately)

    Notes:
        - The function tokenizes inputs by splitting on whitespace and converting to lowercase.
        - Returns 0.0 if either input is empty.
        - No external libraries (e.g., NLTK) should be used to compute the result.
    """

    # BEGIN CODE : eval1.rouge_l
    # ADD YOUR CODE HERE
    def lcs(X, Y):
        """Helper function to compute length of LCS using dynamic programming."""
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m):
            for j in range(n):
                if X[i] == Y[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[m][n]

    if not prediction or not reference:
        return 0.0

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    lcs_len = lcs(pred_tokens, ref_tokens)
    m, n = len(pred_tokens), len(ref_tokens)

    if m == 0 or n == 0 or lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n

    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    return f_score

    # END CODE
## ==== END EVALUATION PORTION

# %% [markdown]
# ### BLEU
# 
# BLEU (Bilingual Evaluation Understudy) is a standard metric for evaluating machine translation quality by comparing the similarity between machine-generated translations and reference translations using n-gram precision.
# 
# It was developed at IBM in 2001 and remains one of the most popular automated metrics for translation evaluation due to its correlation with human judgments.
# 
# Your implementation should follow the original BLEU formulation as described by [Papineni et al. (2002)](https://aclanthology.org/P02-1040/) and summarized on [Wikipedia](https://en.wikipedia.org/wiki/BLEU), with the addition of a smoothing component (as described below).

# %%
## ==== BEGIN EVALUTAION PORTION
def bleu_score(candidate: str, reference: str) -> float:
    """
    Computes the BLEU score for a candidate translation given a reference translation
    using modified n-gram precision up to 4-grams, without applying any smoothing.

    It proceeds as follows:

    1. Tokenize the candidate and reference by converting them to lowercase and splitting on whitespace.
    2. For each n in {1, 2, 3, 4}:
       - Generate n-grams for both the candidate and reference.
       - Count candidate n-grams and clip each count by the maximum count of that n-gram in the reference.
       - If len(candidate n-grams) == 0, set precision_n to 0.
       - Otherwise compute the modified precision:
             If no n-gram matches are found (match_count == 0), use add-1 smoothing:
                 precision_n = (match_count + 1) / (total candidate n-grams + 1)
             Otherwise:
                 precision_n = match_count / total candidate n-grams
    3. Compute the geometric mean of these four precisions. (If any precision is zero after smoothing, the geometric mean will be 0.)
    4. Calculate the brevity penalty (BP):
         - BP = 1 if the candidate’s length is greater than or equal to the reference’s length.
         - Otherwise, BP = exp(1 - (reference_length / candidate_length)).
    5. The final BLEU score is the product of the brevity penalty and the geometric mean of the precisions.


    Parameters:
        candidate (str): The machine-generated translation to evaluate.
        reference (str): The human-created reference translation.

    Returns:
        float: The BLEU score between 0.0 (no match) and 1.0 (perfect match).

    Examples:
        >>> bleu_score("the cat is on the mat", "the cat is on the mat")
        1.0
        >>> bleu_score("the cat is on mat", "the cat is on the mat")
        0.5789  # (approximately)

    Notes:
        - No external libraries (e.g., NLTK) should be used to compute the BLEU score.
    """
    # BEGIN CODE : eval2.bleu_score
    # ADD YOUR CODE HERE
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    
    # Handle empty inputs
    if not candidate_tokens or not reference_tokens:
        return 0.0
    
    # Calculate brevity penalty
    bp = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        bp = math.exp(1 - len(reference_tokens) / len(candidate_tokens))
    
    # Calculate modified n-gram precision for n from 1 to 4
    precisions = []
    for n in range(1, 5):
        if len(candidate_tokens) < n:
            precisions.append(0.0)
            continue
        
        # Count candidate n-grams
        candidate_ngrams = {}
        for i in range(len(candidate_tokens) - n + 1):
            ngram = tuple(candidate_tokens[i:i+n])
            candidate_ngrams[ngram] = candidate_ngrams.get(ngram, 0) + 1
        
        # Count reference n-grams
        reference_ngrams = {}
        for i in range(len(reference_tokens) - n + 1):
            ngram = tuple(reference_tokens[i:i+n])
            reference_ngrams[ngram] = reference_ngrams.get(ngram, 0) + 1
        
        # Calculate clipped count and total
        clipped_count = 0
        for ngram, count in candidate_ngrams.items():
            clipped_count += min(count, reference_ngrams.get(ngram, 0))
        
        total_count = len(candidate_tokens) - n + 1
        
        # Calculate precision with add-1 smoothing if needed
        if clipped_count == 0:
            precision = (clipped_count + 1) / (total_count + 1)
        else:
            precision = clipped_count / total_count
        
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if 0.0 in precisions:
        return 0.0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    
    # Return final BLEU score
    return bp * geometric_mean
    # END CODE
## ==== END EVALUATION PORTION

# %% [markdown]
# ## Attention Head Importance Analysis

# %% [markdown]
# In this section, you'll analyze the importance of attention heads in T5 transformer models across different NLP tasks. The AttentionHeadImportance class provides tools to:
# - Load and preprocess datasets for various NLP tasks (summarization with CNN/DailyMail and translation with WMT16)
# - Compute the importance of attention heads in the T5 model
# - Visualize the importance scores as heatmaps
# 
# The core of this section is implementing the compute_head_importance method, which measures how much each attention head contributes to the model's performance. This method:
# - Takes a dataloader and optional head masks as input
# - Returns importance scores for encoder, decoder, and cross-attention heads
# 
# Your implementation should be flexible enough to work with different T5 variants (t5-small, t5-base, t5-large, etc.) without hardcoding model-specific parameters.
# 
# After computing the head importance scores, you'll visualize them using heatmaps to compare which heads are most critical for different NLP tasks. These insights can be valuable for understanding how transformer models process different types of language tasks, how different attention heads may be important for different tasks and potentially for pruning less important heads to improve efficiency.
# 
# The head importance scores you calculate will be useful for the next part of the assignment, where you'll explore head pruning techniques.
# 

# %% [markdown]
# You can implement the `compute_head_importance` method using any one of the following approaches—or even propose your own method.
# 
# 1. **Gradient-based scoring**:  
#    Compute the gradients of the loss with respect to the head masks (as in [Michel et al. (2019)](https://arxiv.org/abs/1905.10650)). Larger gradient magnitudes indicate that a head has a larger impact on the loss, and hence is more important.
# 
# 2. **Simple ablation**:  
#    Systematically mask (disable) each attention head one at a time and measure the performance drop (e.g., in terms of loss). Heads whose removal causes larger performance degradation are deemed more important.
# 
# 3. **Attention weight magnitude**:  
#    Average the attention weights for each head across samples and time steps. Heads with higher average attention weights might be considered more influential.
# 
# 4. **Confidence analysis**:  
#    Evaluate the sharpness or concentration of each head's attention distribution. Heads that produce less uniform (more confident) distributions may be more specialized and important.
# 
# 5. **Layer-wise relevance propagation (LRP)**:  
#    Use relevance propagation techniques (as in [Voita et al. (2019)](https://arxiv.org/abs/1905.09418)) to trace how much each head contributes to the final output.
# 

# %%
## ==== BEGIN EVALUATION PORTION
class AttentionHeadImportance:
    def __init__(self, model_name="t5-base", device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # BEGIN CODE : ahi.__init__
        # ADD YOUR CODE HERE
        # Get model configuration
        self.config = self.model.config
        self.num_layers = self.config.num_layers
        self.num_heads = self.config.num_heads
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Initialize full attention masks (all 1s) for each attention type
        self.encoder_head_mask = torch.ones(self.num_layers, self.num_heads, device=self.device)
        self.decoder_head_mask = torch.ones(self.num_layers, self.num_heads, device=self.device)
        self.cross_attn_head_mask = torch.ones(self.num_layers, self.num_heads, device=self.device)
        
        # Define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # END CODE

    def get_dataloader(self, path, name=None, split="validation", batch_size=8, shuffle=False):
        dataset = load_dataset(path, name, split=split)
        dataset = self._preprocess_dataset(path, dataset)
        dataset.set_format(type="torch", columns=["input_ids", "labels"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _preprocess_dataset(self, path, dataset):
        if path == "glue":
            return self._preprocess_sst2(dataset)
        elif path == "squad":
            return self._preprocess_squad(dataset)
        elif path == "cnn_dailymail":
            return self._preprocess_cnn_dailymail(dataset)
        elif path == "wmt16":
            return self._preprocess_wmt16(dataset)
        else:
            raise ValueError(f"Preprocessing for dataset {path} is not implemented.")

    def _preprocess_sst2(self, dataset):
        def preprocess(batch):
            source_texts = ["sst2 sentence: " + ex for ex in batch["sentence"]]
            target_texts = ["positive" if label == 1 else "negative" for label in batch["label"]]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=128)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=10)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["sentence", "label", "idx"])
        return dataset

    def _preprocess_squad(self, dataset):
        def preprocess(batch):
            source_texts = ["question: " + q + " context: " + c for q, c in zip(batch["question"], batch["context"])]
            target_texts = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in batch["answers"]]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=128)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=32)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["id", "title", "context", "question", "answers"])
        return dataset

    def _preprocess_cnn_dailymail(self, dataset):
        def preprocess(batch):
            source_texts = ["summarize: " + text for text in batch["article"]]
            target_texts = batch["highlights"]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=512)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=128)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["article", "highlights", "id"])
        return dataset

    def _preprocess_wmt16(self, dataset):
        def preprocess(batch):
            source_texts = ["translate English to German: " + ex["en"] for ex in batch["translation"]]
            target_texts = [ex["de"] for ex in batch["translation"]]
            inputs = self.tokenizer(source_texts, padding="max_length", truncation=True, max_length=128)
            labels = self.tokenizer(target_texts, padding="max_length", truncation=True, max_length=128)
            return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

        dataset = dataset.map(preprocess, batched=True, remove_columns=["translation"])
        return dataset

    def compute_head_importance(self, dataloader, encoder_head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None):
        """
        Computes the importance of attention heads in the T5 model by measuring their influence on model predictions.

        Head importance quantifies how much each attention head contributes to the model's performance/loss.
        Higher importance values indicate heads that have greater impact on the model's predictions.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            A DataLoader containing batches of input data for evaluation.
            Each batch should contain 'input_ids' and 'labels' for the model.

        encoder_head_mask : torch.Tensor or None, optional
            Initial mask for encoder self-attention heads of shape (num_layers, num_heads).
            Contains 0s and 1s, where 1 indicates the head is active and 0 indicates the head is masked/pruned.
            If None, initialized with all heads active (all ones).

        decoder_head_mask : torch.Tensor or None, optional
            Initial mask for decoder self-attention heads of shape (num_layers, num_heads).
            Contains 0s and 1s, where 1 indicates the head is active and 0 indicates the head is masked/pruned.
            If None, initialized with all heads active (all ones).

        cross_attn_head_mask : torch.Tensor or None, optional
            Initial mask for cross-attention heads of shape (num_layers, num_heads).
            Contains 0s and 1s, where 1 indicates the head is active and 0 indicates the head is masked/pruned.
            If None, initialized with all heads active (all ones).

        Returns:
        --------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            A tuple containing three tensors with normalized importance scores:
            1. encoder_head_importance : torch.Tensor of shape (num_layers, num_heads)
            Importance scores for the self-attention heads in the encoder.
            2. decoder_head_importance : torch.Tensor of shape (num_layers, num_heads)
            Importance scores for the self-attention heads in the decoder.a
            3. cross_attn_head_importance : torch.Tensor of shape (num_layers, num_heads)
            Importance scores for the cross-attention heads that connect encoder to decoder.

        Notes:
        ------
        - Head importance is typically measured by examining how each head affects the model's loss.
        - These importance scores can be used for pruning less important heads in the next part of the assignment.
        - The masks, being supplied as additional inputs to this method may make it useful for the pruning task in the next part of the assignment, in case you choose to prune incrementally.
        - Normalize the importance scores to a range of [0, 1] for better visualization and comparison.
        """

        # BEGIN CODE : ahi.compute_head_importance
        # ADD YOUR CODE HERE
        self.model.eval()
        
        # Initialize head masks if not provided
        if encoder_head_mask is None:
            encoder_head_mask = torch.ones((self.num_layers, self.num_heads), device=self.device)
        if decoder_head_mask is None:
            decoder_head_mask = torch.ones((self.num_layers, self.num_heads), device=self.device)
        if cross_attn_head_mask is None:
            cross_attn_head_mask = torch.ones((self.num_layers, self.num_heads), device=self.device)
        
        # First, compute the baseline loss with all heads active
        # print("compute the baseline loss for all active loss")
        # baseline_loss = self._compute_baseline_loss(dataloader, 
        #                                             encoder_head_mask, 
        #                                             decoder_head_mask, 
        #                                             cross_attn_head_mask)
        # print(f'baseline loss: {baseline_loss: .4f}')
        
        # Initialize tensors to store importance scores
        encoder_head_importance = torch.zeros((self.num_layers, self.num_heads), device=self.device)
        decoder_head_importance = torch.zeros((self.num_layers, self.num_heads), device=self.device)
        cross_attn_head_importance = torch.zeros((self.num_layers, self.num_heads), device=self.device)

        # Track number of samples processed
        max_batches = 3
        num_samples = 0
        batch_count = 0

        # Process batches from dataloader
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            for batch in dataloader:
                if batch_count >= max_batches:
                        break
                batch_count += 1
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                batch_size = input_ids.size(0)
                num_samples += batch_size
                
                # Make gradients of head masks require gradients
                encoder_head_mask_grad = encoder_head_mask.clone().requires_grad_(True)
                decoder_head_mask_grad = decoder_head_mask.clone().requires_grad_(True)
                cross_attn_head_mask_grad = cross_attn_head_mask.clone().requires_grad_(True)
                
                # Forward pass with head masks
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    head_mask=encoder_head_mask_grad,
                    decoder_head_mask=decoder_head_mask_grad,
                    cross_attn_head_mask=cross_attn_head_mask_grad,
                )
                
                loss = outputs.loss
                
                # Backward pass to get gradients with respect to head masks
                loss.backward()
                
                # Accumulate importance scores (absolute values of gradients)
                # Higher gradient absolute value = higher importance
                encoder_head_importance += encoder_head_mask_grad.grad.abs().detach() * batch_size
                decoder_head_importance += decoder_head_mask_grad.grad.abs().detach() * batch_size
                cross_attn_head_importance += cross_attn_head_mask_grad.grad.abs().detach() * batch_size
                
                # Clear gradients for next batch
                self.model.zero_grad()

        # Average importance scores over all samples
        if num_samples > 0:
            encoder_head_importance /= num_samples
            decoder_head_importance /= num_samples
            cross_attn_head_importance /= num_samples

        # Normalize importance scores to [0, 1] range for each component
        def normalize(tensor):
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                return (tensor - min_val) / (max_val - min_val)
            return torch.zeros_like(tensor)

        encoder_head_importance = normalize(encoder_head_importance)
        decoder_head_importance = normalize(decoder_head_importance)
        cross_attn_head_importance = normalize(cross_attn_head_importance)

        return encoder_head_importance, decoder_head_importance, cross_attn_head_importance
        # END CODE

    def visualize_head_importance(self, encoder_head_importance, decoder_head_importance, cross_attn_head_importance, save_path="fig.png"):
        fig, axes = plt.subplots(3, 1, figsize=(self.model.config.num_layers, self.model.config.num_heads))

        sns.heatmap(encoder_head_importance.cpu().numpy(), ax=axes[0], cmap="Blues", annot=True, fmt=".2f")
        axes[0].set_title("Encoder Head Importance")
        axes[0].set_xlabel("Heads")
        axes[0].set_ylabel("Layers")

        sns.heatmap(decoder_head_importance.cpu().numpy(), ax=axes[1], cmap="Greens", annot=True, fmt=".2f")
        axes[1].set_title("Decoder Head Importance")
        axes[1].set_xlabel("Heads")
        axes[1].set_ylabel("Layers")

        sns.heatmap(cross_attn_head_importance.cpu().numpy(), ax=axes[2], cmap="Reds", annot=True, fmt=".2f")
        axes[2].set_title("Cross-Attention Head Importance")
        axes[2].set_xlabel("Heads")
        axes[2].set_ylabel("Layers")

        plt.tight_layout()
        plt.savefig(save_path)

    # BEGIN CODE : ahi.additional_methods
    # ADD YOUR CODE HERE
    # You may add additional methods here if needed.
    # END CODE

## ==== END EVALUATION PORTION


# %%
## ==== BEGIN EVALUATION PORTION
# BEGIN CODE : eval_ahi.config
# ADD YOUR CODE HERE
# You are free to change these values to suit your needs
split_cnn_dailymail = "validation[:1%]" # subset of the CNN/DailyMail dataset used for evaluation. "validation[:5%]" uses 5% of the validation set.
batch_size_cnn_dailymail = 2
split_wmt16 = "validation[:1%]"
batch_size_wmt16 = 2
# END CODE
## ==== END EVALUATION PORTION

# %% [markdown]
# ### Evaluate Head Importance on Summarization task.

# %%
# Instantiate the class
ahi = AttentionHeadImportance()

# Load dataset and get dataloader
dataloader = ahi.get_dataloader("cnn_dailymail", "3.0.0", split=split_cnn_dailymail, batch_size=batch_size_cnn_dailymail)

# Compute head importance
encoder_importance, decoder_importance, cross_attn_importance = ahi.compute_head_importance(dataloader)

# Visualize
ahi.visualize_head_importance(encoder_importance, decoder_importance, cross_attn_importance, save_path="head_imp_cnn.png")


# %% [markdown]
# ### Evaluate Head Importance on Translation task.

# %%
# Load dataset and get dataloader
dataloader = ahi.get_dataloader("wmt16", "de-en", split=split_wmt16, batch_size=batch_size_wmt16)
# Compute head importance
encoder_importance, decoder_importance, cross_attn_importance = ahi.compute_head_importance(dataloader)
# Visualize
ahi.visualize_head_importance(encoder_importance, decoder_importance, cross_attn_importance, save_path="head_imp_wmt.png")

# %%
# Load the saved images
img1 = Image.open("head_imp_cnn.png")
img2 = Image.open("head_imp_wmt.png")

# Get dimensions
combined_width = img1.width + img2.width
max_height = max(img1.height, img2.height)

# Create a blank image with the combined width and extra space for labels
combined_img = Image.new("RGB", (combined_width, max_height + 30), (255, 255, 255))  # White background with space for labels

# Paste the images side by side
combined_img.paste(img1, (0, 30))  # Add 30px padding at top for labels
combined_img.paste(img2, (img1.width, 30))

# Add labels
draw = ImageDraw.Draw(combined_img)
font = ImageFont.load_default(size=30)

# Add titles centered above each image
draw.text((img1.width//2, 5), "Summarization (CNN/DailyMail)", fill=(0, 0, 0), font=font, anchor="mt")
draw.text((img1.width + img2.width//2, 5), "Translation (WMT16)", fill=(0, 0, 0), font=font, anchor="mt")

# Draw a subtle separator line
draw.line([(img1.width, 0), (img1.width, max_height + 30)], fill=(200, 200, 200), width=1)

# Show the combined image
display.display(combined_img)

# %% [markdown]
# ### Expected Observations
# 
# When analyzing the attention head importance scores across different tasks like summarization (CNN/DailyMail) and translation (WMT16), you should observe several interesting patterns:
# 
# 1. **Task-Specific Heads**: Some attention heads will show dramatically different importance scores between tasks. For example, certain encoder heads might be crucial for capturing the document structure in summarization but less important for the sentence-level processing in translation. This task-specific specialization suggests that different heads learn to focus on linguistic features relevant to particular tasks.
# 
# 2. **Universally Important Heads**: You'll likely identify a subset of heads with consistently high importance scores across both tasks. These "universal" heads may be responsible for fundamental language understanding capabilities like syntactic parsing or semantic relationship modeling that benefit multiple NLP tasks.
# 

# %% [markdown]
# ## Pruning (Masking) Heads

# %%
## ==== BEGIN EVALUATION PORTION
class Pruner(AttentionHeadImportance):
    def get_mask(self, dataloader, sparsity_to_introduce, remove_from="any", existing_encoder_head_mask=None, existing_decoder_head_mask=None, existing_cross_attn_head_mask=None):
        """
        Creates attention head masks for pruning by removing/masking less important heads based on their importance scores.

        This method identifies which attention heads to prune based on their relative importance
        to the model's performance. Heads with lower importance scores are pruned first.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            A DataLoader containing batches of input data for evaluating head importance.
            Each batch should contain 'input_ids' and 'labels' for the model.

        sparsity_to_introduce : float
            The fraction of total heads to prune (e.g., 0.2 for 20% of heads).
            Must be between 0.0 and 1.0.

        remove_from : str, optional (default="any")
            Which component(s) to prune heads from. Must be one of:
            - "encoder": Only prune encoder self-attention heads
            - "decoder": Only prune decoder self-attention heads
            - "cross": Only prune cross-attention heads
            - "any": Prune from any component based on lowest importance scores

        existing_encoder_head_mask : torch.Tensor or None, optional
            Existing mask for encoder self-attention heads of shape (num_layers, num_heads).
            Contains 0s and 1s, where 1 indicates the head is active and 0 indicates the head is masked/pruned.
            If None, initialized with all heads active (all ones).

        existing_decoder_head_mask : torch.Tensor or None, optional
            Existing mask for decoder self-attention heads of shape (num_layers, num_heads).
            Contains 0s and 1s, where 1 indicates the head is active and 0 indicates the head is masked/pruned.
            If None, initialized with all heads active (all ones).

        existing_cross_attn_head_mask : torch.Tensor or None, optional
            Existing mask for cross-attention heads of shape (num_layers, num_heads).
            Contains 0s and 1s, where 1 indicates the head is active and 0 indicates the head is masked/pruned.
            If None, initialized with all heads active (all ones).

        Returns:
        --------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            A tuple containing three binary mask tensors (0s and 1s):
            1. encoder_head_mask : torch.Tensor of shape (num_layers, num_heads)
            Binary mask for encoder self-attention heads, where 0 indicates pruned heads.
            2. decoder_head_mask : torch.Tensor of shape (num_layers, num_heads)
            Binary mask for decoder self-attention heads, where 0 indicates pruned heads.
            3. cross_attn_head_mask : torch.Tensor of shape (num_layers, num_heads)
            Binary mask for cross-attention heads, where 0 indicates pruned heads.

        Notes:
        ------
        - Utilise compute_head_importance().
        - The resulting masks will be used with the model to evaluate performance with pruned heads.
        - When remove_from="any", remove heads with the lowest importance scores across all components. For example, if sparsity_to_introduce=0.2, then our intention is to remove 20% of the total heads across from any part of the model, from amongst the active heads.
        - If our model has 100 heads in total, and 10 of those are already pruned (as per existing masks), and sparsity_to_introduce=0.2, then we will remove 20% of the original number of heads (i.e., 20% of 100 heads) in the current method call. Hence, the returned masks will have a sparsity of 30% (10 existing + 20 new).
        - When remove_from="encoder", "decoder", or "cross", remove heads only from the specified component. A sparsity of 0.2 means 20% of the heads in that component will be pruned.

        """
        # BEGIN CODE : pruner.get_mask
        # ADD YOUR CODE HERE
       # Initialize existing masks if not provided
        if existing_encoder_head_mask is None:
            existing_encoder_head_mask = torch.ones((self.num_layers, self.num_heads), device=self.device)
        if existing_decoder_head_mask is None:
            existing_decoder_head_mask = torch.ones((self.num_layers, self.num_heads), device=self.device)
        if existing_cross_attn_head_mask is None:
            existing_cross_attn_head_mask = torch.ones((self.num_layers, self.num_heads), device=self.device)
        
        # Create copies of existing masks
        encoder_mask = existing_encoder_head_mask.clone()
        decoder_mask = existing_decoder_head_mask.clone()
        cross_mask = existing_cross_attn_head_mask.clone()
        
        # If no new pruning needed, return existing masks immediately
        if sparsity_to_introduce <= 0:
            return encoder_mask, decoder_mask, cross_mask
        
        # Compute head importance scores using the compute_head_importance method
        # Limit computation by using a single batch
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"][:1].to(self.device)  # Use just one example
        labels = batch["labels"][:1].to(self.device)
        
        # Create a mini-dataloader with just one batch
        mini_batch = {"input_ids": input_ids, "labels": labels}
        mini_dataloader = [mini_batch]
        
        # Compute importance scores
        encoder_importance, decoder_importance, cross_attn_importance = self.compute_head_importance(
            mini_dataloader, 
            encoder_head_mask=encoder_mask,
            decoder_head_mask=decoder_mask,
            cross_attn_head_mask=cross_mask
        )
        
        # Determine which heads to prune based on importance scores
        if remove_from == "any":
            # Count total heads
            total_heads = self.num_layers * self.num_heads * 3  # encoder, decoder, cross
            
            # Calculate number of heads to remove
            heads_to_remove = int(sparsity_to_introduce * total_heads)
            
            # Combine all importance scores
            all_heads = []
            
            # Add encoder heads
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    if encoder_mask[layer, head] > 0:  # Only consider active heads
                        all_heads.append((encoder_importance[layer, head].item(), 'encoder', layer, head))
            
            # Add decoder heads
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    if decoder_mask[layer, head] > 0:
                        all_heads.append((decoder_importance[layer, head].item(), 'decoder', layer, head))
            
            # Add cross-attention heads
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    if cross_mask[layer, head] > 0:
                        all_heads.append((cross_attn_importance[layer, head].item(), 'cross', layer, head))
            
            # Sort by importance (ascending, so least important heads come first)
            all_heads.sort(key=lambda x: x[0])
            
            # Prune the least important heads
            heads_to_remove = min(heads_to_remove, len(all_heads))
            for i in range(heads_to_remove):
                _, component, layer, head = all_heads[i]
                if component == 'encoder':
                    encoder_mask[layer, head] = 0
                elif component == 'decoder':
                    decoder_mask[layer, head] = 0
                else:  # cross
                    cross_mask[layer, head] = 0
        
        elif remove_from == "encoder":
            # Total encoder heads
            total_encoder_heads = self.num_layers * self.num_heads
            
            # Calculate heads to remove
            heads_to_remove = int(sparsity_to_introduce * total_encoder_heads)
            
            # Find encoder heads to prune
            encoder_heads = []
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    if encoder_mask[layer, head] > 0:
                        encoder_heads.append((encoder_importance[layer, head].item(), layer, head))
            
            # Sort by importance
            encoder_heads.sort(key=lambda x: x[0])
            
            # Prune heads
            heads_to_remove = min(heads_to_remove, len(encoder_heads))
            for i in range(heads_to_remove):
                _, layer, head = encoder_heads[i]
                encoder_mask[layer, head] = 0
        
        elif remove_from == "decoder":
            # Total decoder heads
            total_decoder_heads = self.num_layers * self.num_heads
            
            # Calculate heads to remove
            heads_to_remove = int(sparsity_to_introduce * total_decoder_heads)
            
            # Find decoder heads to prune
            decoder_heads = []
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    if decoder_mask[layer, head] > 0:
                        decoder_heads.append((decoder_importance[layer, head].item(), layer, head))
            
            # Sort by importance
            decoder_heads.sort(key=lambda x: x[0])
            
            # Prune heads
            heads_to_remove = min(heads_to_remove, len(decoder_heads))
            for i in range(heads_to_remove):
                _, layer, head = decoder_heads[i]
                decoder_mask[layer, head] = 0
        
        elif remove_from == "cross":
            # Total cross-attention heads
            total_cross_heads = self.num_layers * self.num_heads
            
            # Calculate heads to remove
            heads_to_remove = int(sparsity_to_introduce * total_cross_heads)
            
            # Find cross-attention heads to prune
            cross_heads = []
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    if cross_mask[layer, head] > 0:
                        cross_heads.append((cross_attn_importance[layer, head].item(), layer, head))
            
            # Sort by importance
            cross_heads.sort(key=lambda x: x[0])
            
            # Prune heads
            heads_to_remove = min(heads_to_remove, len(cross_heads))
            for i in range(heads_to_remove):
                _, layer, head = cross_heads[i]
                cross_mask[layer, head] = 0
        
        return encoder_mask, decoder_mask, cross_mask
        # END CODE
## ==== END EVALUATION PORTION

# %% [markdown]
# ### EVALUATION OF THE PRUNING STRATEGY

# %%

def evaluate_at_different_sparsities(pruner, train_dataloader, test_dataloader, sparsities=[0, 0.2, 0.4, 0.6, 0.8, 1.0], remove_from="any"):

    encoder_mask = torch.ones((pruner.model.config.num_layers, pruner.model.config.num_heads),
                                         device=pruner.device)
    decoder_mask = torch.ones((pruner.model.config.num_layers, pruner.model.config.num_heads),
                                         device=pruner.device)
    cross_mask = torch.ones((pruner.model.config.num_layers, pruner.model.config.num_heads),
                                         device=pruner.device)

    bleu_scores = []
    cur_sparsity = 0

    for sparsity in sparsities:

        sparsity_to_introduce = sparsity - cur_sparsity
        encoder_mask, decoder_mask, cross_mask = pruner.get_mask(train_dataloader, sparsity_to_introduce, remove_from=remove_from, existing_encoder_head_mask=encoder_mask, existing_decoder_head_mask=decoder_mask, existing_cross_attn_head_mask=cross_mask)

        cur_sparsity = sparsity
        print(f"Performing evaluation when sparsity={sparsity}")
        cur_bleu_scores = []

        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(pruner.device)
            labels = batch["labels"].to(pruner.device)

            output_ids = pruner.model.generate(
                input_ids,
                head_mask=encoder_mask,
                decoder_head_mask=decoder_mask,
                cross_attn_head_mask=cross_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )

            for i in range(len(input_ids)):
                prompt = pruner.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                reference = pruner.tokenizer.decode(labels[i], skip_special_tokens=True)
                candidate = pruner.tokenizer.decode(output_ids[i], skip_special_tokens=True)
                cur_bleu_scores.append(bleu_score(candidate, reference))

        average_bleu = sum(cur_bleu_scores) / len(cur_bleu_scores)
        bleu_scores.append(average_bleu)
    return bleu_scores

# %%
torch.manual_seed(42)

pruner = Pruner(model_name="t5-base")
dataloader_wmt = pruner.get_dataloader("wmt16", "de-en", split="validation[:1%]", batch_size=5) # this dataloader will be used for calculating the masks
dataloader_wmt_test = pruner.get_dataloader("wmt16", "de-en", split="test[:1%]", batch_size=5)

# %% [markdown]
# #### Pruning heads from any part of the model

# %%
torch.manual_seed(42)

sparsities = [0, 0.2, 0.4, 0.6, 0.8, 1]

bleu_scores = evaluate_at_different_sparsities(pruner, dataloader_wmt, dataloader_wmt_test, sparsities, remove_from="any")

# Print results in a formatted table
print("\nSparsity vs BLEU Score")
print("=" * 30)
for sparsity, score in zip(sparsities, bleu_scores):
    print(f"Sparsity: {sparsity:.1f}  |  BLEU Score: {score:.4f}")

# Plot BLEU Score vs Sparsity
plt.figure(figsize=(7, 5))
plt.plot(sparsities, bleu_scores, marker='o', linestyle='-', color='orange', label="Pruning heads from any part of the model")
plt.xlabel("Sparsity Level")
plt.ylabel("BLEU Score")
plt.title("BLEU Score vs Sparsity")
plt.ylim(0, 0.40)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(sparsities)
plt.savefig('prune_any.png')

score_no_sparsity = bleu_scores[0]

# %% [markdown]
# ### Expected Observations
# 
# The bleu score will decrease as the sparsity increases but you should observe that using carefully curated masks would result in atleast 20-30% better bleu scores than random masking for 50% sparsity. As the degree of sparsity increases, the relative performance improvement should increase significantly.

# %% [markdown]
# #### Pruning heads from individual components

# %%
torch.manual_seed(42)
sparsities = [0.2, 0.4, 0.6, 0.8, 1.0]

print("Masking heads in the encoder layers only...\n")
encoder_bleu_scores = evaluate_at_different_sparsities(pruner, dataloader_wmt, dataloader_wmt_test, sparsities, remove_from="encoder")

print("\nEncoder Sparsity vs BLEU Score")
print("=" * 30)

for sparsity, score in zip(sparsities, encoder_bleu_scores):
    print(f"Sparsity: {sparsity:.1f}  |  BLEU Score: {score:.4f}")

torch.manual_seed(42)

print("\nMasking heads in the decoder layers only...\n")
decoder_bleu_scores = evaluate_at_different_sparsities(pruner, dataloader_wmt, dataloader_wmt_test, sparsities, remove_from="decoder")

print("\nDecoder Sparsity vs BLEU Score")
print("=" * 30)

for sparsity, score in zip(sparsities, decoder_bleu_scores):
    print(f"Sparsity: {sparsity:.1f}  |  BLEU Score: {score:.4f}")

torch.manual_seed(42)

print("\nMasking cross-attention heads only...\n")
cross_bleu_scores = evaluate_at_different_sparsities(pruner, dataloader_wmt, dataloader_wmt_test, sparsities, remove_from="cross")

print("\nCross-Attention Sparsity vs BLEU Score")
print("=" * 30)
for sparsity, score in zip(sparsities, cross_bleu_scores):
    print(f"Sparsity: {sparsity:.1f}  |  BLEU Score: {score:.4f}")

# Prepend and append the scores for no sparsity and total sparsity
encoder_bleu_scores.insert(0, score_no_sparsity)

decoder_bleu_scores.insert(0, score_no_sparsity)

cross_bleu_scores.insert(0, score_no_sparsity)

sparsities.insert(0, 0)

plt.figure(figsize=(7, 5))
plt.plot(sparsities, encoder_bleu_scores,
         marker='o', linestyle='-', color='blue', label="Pruning Encoder Heads")
plt.plot(sparsities, decoder_bleu_scores,
         marker='o', linestyle='-', color='green', label="Pruning Decoder Heads")
plt.plot(sparsities, cross_bleu_scores,
         marker='o', linestyle='-', color='orange', label="Pruning Cross-Attention Heads")
plt.xlabel("Sparsity Level")
plt.ylabel("BLEU Score")
plt.title("BLEU Score vs Sparsity (Encoder only, Decoder only, Cross-Attention only)")
plt.ylim(0, 0.40)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(sparsities)
plt.savefig('prune_encoder_decoder_cross.png')

# %% [markdown]
# ### Expected Observations
# 
# You should see that the impact of pruning to different sparsity levels in different components (encoder, decoder, cross-attention) is not the same. Pruning a certain percentage from some component may result in a much larger performance degradation compared to pruning the same percentage from another component. Moreover, you might observe that when you prune carefully, the performance degradation from component A might be worse than that for component B. But when you prune randomly, the trend might switch. This may mean that certain components contain a higher concentration of critical heads whilst other components have their importance more evenly distributed across heads, making them more resilient to targeted pruning but potentially more vulnerable to random pruning.

# %% [markdown]
# # Congratulations!
# 
# You've reached the end of the assignment. Hope this was a pleasant and fun exercise!
# 
# Check the submission instructions mentioned at the beginning of the notebook in order to submit your homework.
# 


