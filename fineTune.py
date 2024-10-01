from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

# set the pad_token to eos_token or add a new pad token
tokenizer.pad_token = tokenizer.eos_token

# load the dataset
dataset = load_from_disk("code_generation_dataset")

# split the dataset into training and test sets
dataset = dataset.train_test_split(test_size=0.1)

# preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['code'], truncation=True, padding='max_length')