# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ilsilfverskiold/tech-keywords-extractor")
model = AutoModelForSeq2SeqLM.from_pretrained("ilsilfverskiold/tech-keywords-extractor")

ARTICLE_TO_SUMMARIZE = (
    "A binary search tree (BST) is a type of data structure where each node has up to two children, named the left and right children. In a BST, the left child’s value is less than its parent node’s value, and the right child’s value is greater than its parent node’s value. This arrangement enables quick search, insert, and delete operations."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="pt")

summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)