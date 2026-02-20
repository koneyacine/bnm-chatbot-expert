# NOTE: This script requires a GPU (NVIDIA) and the 'unsloth' library.
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# 1. Configuration
model_name = "unsloth/Qwen2.5-14B-bnb-4bit" # Passage à la version 14B pour plus de performance
max_seq_length = 2048
dataset_file = "bnm_dataset.jsonl"

def fine_tune():
    # 2. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    # 3. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # 4. Load & Split Dataset (80% Train, 20% Eval)
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 5. Trainer Configuration
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "messages",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 500, # Augmenté à 500 comme demandé
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            eval_steps = 50,
            evaluation_strategy = "steps",
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "lora_model",
        ),
    )

    # 6. Start Training
    print("Démarrage du Fine-Tuning LoRA (Qwen 14B)...")
    trainer.train()

    # 7. Save Model (Merge to 4-bit)
    print("Sauvegarde et merge du modèle...")
    model.save_pretrained_merged("bnm_model_lora", tokenizer, save_method = "merged_4bit")
    print("Terminé !")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Erreur : Aucun GPU NVIDIA détecté. Le fine-tuning LoRA n'est pas possible sans GPU.")
    else:
        fine_tune()
