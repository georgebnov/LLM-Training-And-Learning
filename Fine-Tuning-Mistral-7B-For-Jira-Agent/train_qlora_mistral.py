import os, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from trl import SFTConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "mistral7b-qlora-jira"
DATASET = "jira_finetune_data.jsonl"

#Quantization fo the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files=DATASET, split="train")
dataset = dataset.rename_column("prompt", "text")
splits = dataset.train_test_split(test_size=0.1, seed=42)   # 90/10 split, 90 training 10 testing
train_data = splits["train"]
eval_data  = splits["test"]

def formatting_func(ex):
    return ex["text"]

# Optional: quick check
print("Sample training text:\n", train_data[0]["text"])

#RTX3080 this should be the best 
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=10,
    max_steps=-1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,
    eval_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    report_to=[], 
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    weight_decay=0.0,
    dataloader_num_workers=2,
    max_length=4096,
    packing=True,
    completion_only_loss=False,
    do_eval=True,     
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config, 
    args=sft_config,
    processing_class=tokenizer,  
    formatting_func=formatting_func,
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR) #LoRA adapters are saved
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
