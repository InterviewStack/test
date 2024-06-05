from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import threading
import time

def gemma(prompt):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.bfloat16)

    input_text = prompt
    input_ids = tokenizer(input_text, return_tensors="pt")

    start_time = time.time()
    outputs = model.generate(**input_ids, max_length=500)
    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")

    return tokenizer.decode(outputs[0])

model_answer = "Model Answer: A binary search tree (BST) is a data structure in which each node has at most two children referred to as the left child and the right child. For each node, the left subtree contains only nodes with keys less than the node’s key, and the right subtree contains only nodes with keys greater than the node’s key. This property allows for efficient searching, insertion, and deletion operations."
user_answer = " User Answer: A binary search tree (BST) is a type of data structure where each node has up to two children, named the left and right children. In a BST, the left child’s value is less than its parent node’s value, and the right child’s value is greater than its parent node’s value. This arrangement enables quick search, insert, and delete operations."
context = "Context: You are an AI with the task of comparing 2 answers for an interview, a model answewr and user's answer. Give me an analysis with a correctness score and some advice for the user if required. "
analysis = gemma(context+model_answer+user_answer)
print(analysis)