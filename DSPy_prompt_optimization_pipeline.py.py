"""
This script demonstrates a complete pipeline for configuring a local AI model using DSPy and Ollama,
loading a labeled dataset for training, optimizing the model with hints, and using the optimized model 
for text classification. It uses the Banking77 dataset and showcases the following steps:
1. Model configuration with DSPy and a local Ollama API.
2. Loading and preparing a labeled dataset with hints for classification.
3. Defining a DSPy module for classification tasks.
4. Optimizing the module using BootstrapFinetune to improve performance.
5. Saving the optimized program for later use.
6. Testing the optimized classifier with a sample input.
"""

import random
from typing import Literal
from dspy.datasets import DataLoader
from datasets import load_dataset
import dspy
import litellm

# Enable LiteLLM parameter dropping
litellm.drop_params = True

# Configure the local model using Ollama
print("Configuring local model using Ollama...")
lm = dspy.LM('ollama_chat/artifish/llama3.2-uncensored:latest', api_base='http://192.168.1.155:11434', api_key='')
dspy.configure(lm=lm)
print("Model configured successfully.")

# Load the Banking77 dataset
print("Loading the Banking77 dataset...")
CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
print(f"Classes loaded: {CLASSES}")
kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

# Load the first 2000 examples from the dataset and assign a hint to each training example
print("Preparing the training dataset...")
trainset = [
    dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
]
random.Random(0).shuffle(trainset)
print(f"Training dataset prepared with {len(trainset)} examples.")

# Define the DSPy module for classification with signature
print("Defining the DSPy classification module...")
signature = dspy.Signature("text -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
classify = dspy.ChainOfThoughtWithHint(signature)
print("DSPy classification module defined.")

# Optimize via BootstrapFinetune
print("Optimizing the model using BootstrapFinetune...")
dspy.settings.experimental = True
optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
optimized = optimizer.compile(classify, trainset=trainset)
print("Optimization completed successfully.")
print("Optimized model:", optimized)

# Save the optimized model
print("Saving the optimized program...")
optimized.save("optimized_program.json")
print("Optimized program saved as 'optimized_program.json'.")

# Run the optimized classifier on a sample input
print("Running the optimized classifier on a sample input...")
result = optimized_classifier(text="What does a pending cash withdrawal mean?")
print("Result:", result)
