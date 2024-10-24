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

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()

# define a function to generate code using the fine-tuned model
def generate_code(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# test the model with a code generation prompt
prompt = "def merge_sort(arr):"
generated_code = generate_code(prompt)

print("Generated Code:")
print(generated_code)