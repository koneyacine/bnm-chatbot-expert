# NOTE: This script requires a GPU (NVIDIA) and the 'unsloth' library.
# To install: pip install unsloth[colab-new]
# Or see: https://github.com/unslothai/unsloth

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Configuration
model_name = "unsloth/Qwen2.5-7B-bnb-4bit" # Version 4-bit pour économiser la VRAM
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
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # 4. Load Dataset
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # 5. Trainer Configuration
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "messages", # Unsloth gère automatiquement le format ChatML
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Ajustable selon le nombre d'époques souhaitées
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "lora_model",
        ),
    )

    # 6. Start Training
    print("Démarrage du Fine-Tuning LoRA...")
    trainer.train()

    # 7. Save Model
    print("Sauvegarde du modèle fine-tuné...")
    model.save_pretrained_merged("bnm_model_lora", tokenizer, save_method = "merged_16bit")
    print("Terminé !")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Erreur : Aucun GPU NVIDIA détecté. Le fine-tuning LoRA n'est pas possible sans GPU.")
    else:
        fine_tune()
