# CUDA_VISIBLE_DEVICES=6,7,8,9 python cpt_sft_ex_multi_gpu.py
import torch
import datetime
import os
now = datetime.datetime.now().strftime('%m%d')
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling, 
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets, Dataset

# 配置參數
model_name = "google/gemma-3-12b-it"
cache_dir = "/mnt/usr1/azure005/LLM/models"
max_seq_length = 2880  # 定義最大序列長度

# ❌ 移除量化配置 - 使用 BF16 原生精度
# 載入模型
print("Loading model and tokenizer...")
print("🚀 多 GPU 訓練模式 - 不使用量化")
# 自動找到本地模型路徑
import glob
model_snapshot_path = glob.glob("/mnt/usr1/azure005/LLM/gemma-3-12b-it")[0]
print(f"Using local model from: {model_snapshot_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_snapshot_path,
    device_map="auto",  # 自動分配到多張 GPU
    dtype=torch.bfloat16,  # 使用 BFloat16 精度
    torch_dtype=torch.bfloat16,  # 向後兼容
    trust_remote_code=True,
    # use_cache 參數在載入時不需要，稍後在 config 中設定
)

# ✅ 設定 cache 和其他訓練配置
model.config.use_cache = False  # 訓練時必須關閉 cache

# ✅ 啟用梯度檢查點
model.gradient_checkpointing_enable()

# LoRA 配置
print("Setting up LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ✅ 確保 LoRA 參數可訓練
for name, param in model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True
model.config.pretraining_tp = 1
model.config.use_cache = False

# 顯示可訓練參數
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_snapshot_path,
    trust_remote_code=True,
    padding_side="right",
    add_eos_token=True,
    add_bos_token=True
)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

class CustomDataCollatorForCausalLM:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        import torch
        # 提取 input_ids（可能是列表）
        input_ids = [torch.tensor(f["input_ids"]) if not isinstance(f["input_ids"], torch.Tensor) else f["input_ids"] for f in features]
        
        # 找出最大長度
        max_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Padding
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for ids in input_ids:
            # 計算需要 padding 的長度
            padding_length = max_length - len(ids)
            
            # Pad input_ids
            padded_ids = torch.cat([
                ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
            ])
            batch_input_ids.append(padded_ids)
            
            # 創建 attention_mask
            attention_mask = torch.cat([
                torch.ones(len(ids), dtype=torch.long),
                torch.zeros(padding_length, dtype=torch.long)
            ])
            batch_attention_mask.append(attention_mask)
            
            # 創建 labels（padding 位置設為 -100）
            labels = torch.cat([
                ids.clone(),
                torch.full((padding_length,), -100, dtype=ids.dtype)
            ])
            batch_labels.append(labels)
        
        # 堆疊成 batch
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_mask = torch.stack(batch_attention_mask)
        batch_labels = torch.stack(batch_labels)
        
        # 處理 EOS token 的特殊邏輯
        for i in range(batch_labels.shape[0]):
            eos_positions = (batch_input_ids[i] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                last_eos_pos = eos_positions[-1]
                batch_labels[i, last_eos_pos] = batch_input_ids[i, last_eos_pos]
                if len(eos_positions) > 1:
                    second_last_eos = eos_positions[-2]
                    batch_labels[i, second_last_eos] = batch_input_ids[i, second_last_eos]
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }

data_collator = CustomDataCollatorForCausalLM(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)

# CPT
print("Loading and preprocessing CPT dataset...")
# 使用正確的資料路徑
data_dir = "/mnt/usr1/azure005/LLM/training_data/CPT"
moe = load_dataset("json", data_files=f"{data_dir}/教育部臺灣台語常用詞辭典/moedict.json", split="train")
ots = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-ots.json", split="train")
tks = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-tks.json", split="train")
abt   = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-abt.json", split="train")
kkh   = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-kkh.json", split="train")
olbt  = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-olbt.json", split="train")
ljk   = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-ljk.json", split="train")
tsk   = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-tsk.json", split="train")
manlajo = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-manlajo.json", split="train")
asts = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-asts.json", split="train")
achiak = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-achiak.json", split="train")
ngkh = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-ngkh.json", split="train")
ttshs = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-ttshs.json", split="train")
pikh = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-pikh.json", split="train")
khg = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-khg.json", split="train")
sslts = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-sslts.json", split="train")
lgs = load_dataset("json", data_files=f"{data_dir}/IMA-Taiwan-taigi-literature/taigi-literature-lgs.json", split="train")
dataset = concatenate_datasets([moe, ots,tks, sslts, ljk, kkh, tsk, olbt, abt, manlajo, asts, achiak, ngkh, ttshs, pikh, khg, lgs]) #63772

wikipedia_prompt = """Taiwanese Literature
### Title: {}

### Article:
{}"""

def formatting_prompts_func(examples):
    titles = examples["title"]
    texts = examples["text"]
    outputs = []
    for title, text in zip(titles, texts):
        formatted_text = wikipedia_prompt.format(title, text) + tokenizer.eos_token
        outputs.append(formatted_text)
    return {"text": outputs}

dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=4)
dataset = dataset.shuffle(seed=42)

# Tokenization 函數
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # 保持 False，讓 data collator 動態 padding
    )
    return tokenized

# 處理資料集
dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names, 
    num_proc=4
)

# 清理記憶體
torch.cuda.empty_cache()

# 訓練參數 - 多 GPU 不量化優化 (降低記憶體使用)
training_args = TrainingArguments(
    output_dir=f"models/output_gemma3_12b_{now}",
    per_device_train_batch_size=1,  # ✅ 降低至 1 避免 OOM
    gradient_accumulation_steps=8,  # ✅ 增加累積步數維持有效 batch size (4 GPUs × 1 × 8 = 32)
    learning_rate=4e-5,
    bf16=True,
    fp16=False,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=5,
    report_to="wandb",
    run_name=f"gemma3_12b_cpt_{now}",
    logging_steps=10,
    seed=3407,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    optim="adamw_torch",  # ✅ 不量化時用標準 AdamW
    gradient_checkpointing=True,  # ✅ 啟用梯度檢查點以節省記憶體
    max_grad_norm=1.0, # 梯度裁剪
    ddp_find_unused_parameters=False,
    disable_tqdm=False,
    save_only_model=True # 是checkpoint 的模型要不可續訓，較省硬碟空間
    # DDP 會自動啟用
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 開始訓練
print("CPT starting training...")
print(f"✅ 有效 batch size: {training_args.per_device_train_batch_size} × {torch.cuda.device_count()} × {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * torch.cuda.device_count() * training_args.gradient_accumulation_steps}")
trainer_stats = trainer.train()

# 儲存最終模型
trainer.save_model(f"models/final_gemma3_12b_cpt_multi_gpu_{now}")
tokenizer.save_pretrained(f"models/final_gemma3_12b_cpt_multi_gpu_{now}")
print("CPT training completed!")
print(f"CPT training stats: {trainer_stats}")
import wandb
wandb.finish()



# SFT
print("\n" + "="*50)
print("Loading and preprocessing SFT dataset...")
print("="*50)
sft_data_dir = "/mnt/usr1/azure005/LLM/training_data/SFT"
alpaca_dataset = load_dataset("json", data_files=f"{sft_data_dir}/alpaca_translated_dataset_taigi_zh_tw.json", split="train") #48818
tech_dataset = load_dataset("json", data_files=f"{sft_data_dir}/科技大擂台訓練集文本_taigi/AI_tw-2.json", split="train") #13550

# 載入 taigi_parallel_corpus 資料集
print("Loading taigi_parallel_corpus...")
parallel_corpus_dir = f"{sft_data_dir}/taigi_parallel_corpus/訓練用語料"

def load_parallel_data(src_file, trg_file):
    """載入平行語料"""
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines()]
    with open(trg_file, 'r', encoding='utf-8') as f:
        trg_lines = [line.strip() for line in f.readlines()]
    return list(zip(src_lines, trg_lines))

# 載入所有平行語料（選擇最有用的 zh2tw 和 tw2py）
zh2tw_orig1 = load_parallel_data(
    f"{parallel_corpus_dir}/zh2tw/orig1/train.tags.src",
    f"{parallel_corpus_dir}/zh2tw/orig1/train.tags.trg"
)
zh2tw_orig2 = load_parallel_data(
    f"{parallel_corpus_dir}/zh2tw/orig2/train.tags.src",
    f"{parallel_corpus_dir}/zh2tw/orig2/train.tags.trg"
)
tw2py_orig1 = load_parallel_data(
    f"{parallel_corpus_dir}/tw2py/orig1/train.tags.src",
    f"{parallel_corpus_dir}/tw2py/orig1/train.tags.trg"
)
tw2py_orig2 = load_parallel_data(
    f"{parallel_corpus_dir}/tw2py/orig2/train.tags.src",
    f"{parallel_corpus_dir}/tw2py/orig2/train.tags.trg"
)

print(f"✅ Loaded zh2tw: {len(zh2tw_orig1) + len(zh2tw_orig2):,} pairs")
print(f"✅ Loaded tw2py: {len(tw2py_orig1) + len(tw2py_orig2):,} pairs")

converted_list = []

# 1. 處理 Alpaca 資料集
print("Processing Alpaca dataset...")
for example in alpaca_dataset:
    instruction = example.get("instruction", "").strip()
    input_text  = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    user_content = (instruction + "\n" + input_text).strip()
    conversation = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output_text}
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    converted_list.append({"text": text})
print(f"✅ Alpaca: {len(converted_list):,} samples")

import random

# 2. 處理科技大擂台資料集
print("Processing Tech dataset...")
def create_diverse_prompts(instruction, input_text, output_text):
    simple_templates = [
        f"Please select the correct answer.\n\n{input_text}",
        f"Choose the right option from the following.\n\n{input_text}",
        f"Select the correct answer from the given question and choices.\n\n{input_text}",
        f"Answer the following question.\n\n{input_text}",
        f"Please choose the correct option.\n\n{input_text}",
        f"Please select the most appropriate answer from the options below.\n\n{input_text}",
    ]
    user_content = random.choice(simple_templates)
    assistant_content = output_text
    return user_content, assistant_content

tech_start = len(converted_list)
for example in tech_dataset:
    Instruction = example.get("Instruction", "").strip()
    input_text  = example.get("Input", "").strip()
    output_text = example.get("Output", "").strip()
    user_content, assistant_content = create_diverse_prompts(
        Instruction, input_text, output_text
    )
    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    converted_list.append({"text": text})
print(f"✅ Tech: {len(converted_list) - tech_start:,} samples")

# 3. 處理 taigi_parallel_corpus - zh2tw (中文翻譯成台文)
print("Processing zh2tw parallel corpus...")
zh2tw_templates = [
    "請將以下中文翻譯成台文：\n\n{}",
    "請翻譯成台文：\n\n{}",
    "請用台文表達以下內容：\n\n{}",
    "翻譯成台文：\n\n{}",
    "請將下列中文轉換為台文：\n\n{}",
]
zh2tw_start = len(converted_list)
for src, trg in zh2tw_orig1 + zh2tw_orig2:
    # 去除空格還原文本
    src_text = src.replace(" ", "")
    trg_text = trg.replace(" ", "")
    
    if len(src_text) < 5 or len(trg_text) < 5:  # 過濾過短的資料
        continue
    
    user_content = random.choice(zh2tw_templates).format(src_text)
    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": trg_text}
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    converted_list.append({"text": text})
print(f"✅ zh2tw: {len(converted_list) - zh2tw_start:,} samples")

# 4. 處理 taigi_parallel_corpus - tw2py (台文轉拼音)
print("Processing tw2py parallel corpus...")
tw2py_templates = [
    "請將以下台文標注台羅拼音：\n\n{}",
    "請為下列台文加上台羅拼音標記：\n\n{}",
    "請用台羅拼音表示：\n\n{}",
    "請標注台羅拼音：\n\n{}",
    "請將這段台文轉換成台羅拼音：\n\n{}",
]
tw2py_start = len(converted_list)
for src, trg in tw2py_orig1 + tw2py_orig2:
    # 去除空格還原文本
    src_text = src.replace(" ", "")
    trg_text = trg.replace(" ", "")
    
    if len(src_text) < 5 or len(trg_text) < 5:  # 過濾過短的資料
        continue
    
    user_content = random.choice(tw2py_templates).format(src_text)
    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": trg_text}
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    converted_list.append({"text": text})
print(f"✅ tw2py: {len(converted_list) - tw2py_start:,} samples")

print(f"\n📊 Total SFT samples: {len(converted_list):,}")

dataset = Dataset.from_list(converted_list)
print(f"✅ Created dataset with {len(dataset):,} samples")
dataset = dataset.shuffle(seed=42)

# 資料預處理函數
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_seq_length,
        truncation=True,
        padding=False,
    )
    return model_inputs

# 應用預處理
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=4,
)

# 設定訓練參數 - SFT 階段 (降低記憶體使用)
training_args = TrainingArguments(
    output_dir=f"models/output_gemma3_12b_sft_{now}",
    per_device_train_batch_size=1,  # ✅ 降低至 1 避免 OOM
    gradient_accumulation_steps=8,  # ✅ 增加累積步數維持有效 batch size
    learning_rate=4e-5,
    bf16=True,
    fp16=False,
    num_train_epochs=1, 
    save_steps=100,
    save_total_limit=5,
    report_to="wandb",
    run_name=f"gemma3_12b_sft_{now}",
    logging_steps=10,
    seed=3407,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01, #L2正則化 #LoRA或QLoRA通常不需要太高的weight decay
    optim="adamw_torch",
    gradient_checkpointing=True,  # ✅ 啟用梯度檢查點以節省記憶體
    max_grad_norm=1.0,
    ddp_find_unused_parameters=False,
    disable_tqdm=False,
    save_only_model=True # 是checkpoint 的模型要不可續訓，較省硬碟空間
)

# 建立訓練器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# 開始訓練
print("SFT starting training...")
print(f"✅ 有效 batch size: {training_args.per_device_train_batch_size} × {torch.cuda.device_count()} × {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * torch.cuda.device_count() * training_args.gradient_accumulation_steps}")
trainer_stats = trainer.train()

# 儲存模型
print("SFT saving model...")
trainer.save_model(f"models/final_gemma3_12b_sft_{now}")
tokenizer.save_pretrained(f"models/final_gemma3_12b_sft_{now}")
print("\n" + "="*50)
print("🎉 SFT training completed!")
print("="*50)
print(f"SFT training stats: {trainer_stats}")

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
    
print(f"\n✅ All training completed! Models saved in /mnt/usr1/azure005/LLM/models/")
print(f"📊 CPT Model: models/final_gemma3_12b_cpt_{now}")
print(f"📊 SFT Model: models/final_gemma3_12b_sft_{now}")
