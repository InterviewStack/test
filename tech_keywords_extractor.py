# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ilsilfverskiold/tech-keywords-extractor")
model = AutoModelForSeq2SeqLM.from_pretrained("ilsilfverskiold/tech-keywords-extractor")

ARTICLE_TO_SUMMARIZE = (
    "A binary search tree (BST) is a data structure in which each node has at most two children referred to as the left child and the right child. For each node, the left subtree contains only nodes with keys less than the node’s key, and the right subtree contains only nodes with keys greater than the node’s key. This property allows for efficient searching, insertion, and deletion operations."
    "A binary search tree (BST) is a data structure in which each node has at most two children referred to as the left child and the right child. For each node, the left subtree contains only nodes with keys less than the node’s key, and the right subtree contains only nodes with keys greater than the node’s key. This property allows for efficient searching, insertion, and deletion operations."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="pt")

summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)