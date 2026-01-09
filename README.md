# LLM-TW

### Fine-tuning Gemma-3-12B-it for Taiwanese Language Understanding and Reasoning

## 📁 Project Structure
```
├── gemma-3-12b-it/
│   ├── added_tokens.json                 
│   ├── chat_template.jinja  
│   ├── config.json   
│   └── ...
├── models/
│   ├── final_gemma-3-12b_cpt_1022    
│   ├── final_gemma-3-12b_spt_1022
│   ├── output_gemma3_12b_cft_1028                  
│   └── output_gemma3_12b_sft_1028 
├── training_data/
│   ├── CPT            
│   │   ├── IMA-Taiwan-taigi-literature
│   │   │   └── ... 
│   │   └── 教育部臺灣台語常用詞辭典
│   │       └── ... 
│   └── SFT   
│       ├── taigi_parallel_corpus
│       │   └── ... 
│       ├── 科技大擂台訓練集文本_taigi
│       │   └── ... 
│       └── alpaca_translated_dataset_taigi_zh_tw.json
├── wandb/                                                       
│   └── ...  
├── test_data/                                                       
│   └── ...  
├── output_result/                                                       
│   └── ... 
├── cpt_sft_ex.py
├── download_gemma.py
├── README.md
├── test
├── train.sh    
├── gemma3_12b_cpt_sft.log           
└── output_gemma3_12b_sft_1028  

```

## 1️⃣ Environment Setup
#### Recommended Hardware

- Hardware specifications recommendation: NVIDIA 4090 GPU

#### Miniconda Installation

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

```

#### Create and Activate Environment
```bash
conda create -n myenv python=3.10 # "myenv": you can choose the name of environment you like.
conda activate myenv
cd LLM # enter the project folder
pip install -r requirements.txt

```

## 2. 2️⃣ Download Gemma Model

This script automatically downloads the Gemma-3-12B-it model and tokenizer from Hugging Face:

```bash
python download_gemma.py
```
Modify the following parameters in `download_gemma.py`:

- `model_name` → specify which Gemma variant you want (e.g., gemma-3-1b-it).

- `local_dir` → specify where to save the model locally.

## 3️⃣ Weights & Biases (W&B) Setup
#### Login

```bash
wandb login
```
1. Create a W&B account at https://wandb.ai/authorize

2. Copy your personal API key

3. Paste it into your terminal after running wandb login

Once logged in, W&B automatically tracks your training progress (loss, LR, GPU usage, etc.) when you run the training script.

## 4️⃣ Train the Model

- Make the shell script executable and launch training:

```bash
chmod +x train.sh
./train.sh

```
- You can also adjust which GPUs to use by editing:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
```
inside `train.sh`.

- To monitor training progress:

```bash
tail -f gemma3_12b_cpt_sft.log

```
- Replace the filename if your log file differs.

- Logs are stored automatically in your project root directory.

- You can also view real-time metrics on Weights & Biases (wandb).

If existing problem, you should go back to resolve, then

```bash
nvidia-smi   # find the process ID (PID)
kill <PID>   # stop the training process

``` 

## 5️⃣ Test the Model
#### Run model evaluation:

```bash
python test.py \
  --model_path ./models/final_gemma3_12b_sft_1103 \
  --test_file ./data/test_questions.json \
  --output_file ./results/result_1103.json

```

#### Test Dataset Format

```
[
  {
    "題號": "1",
    "題目": "台語的「食」是什麼意思？",
    "選項1": "睡覺",
    "選項2": "吃飯",
    "選項3": "玩耍",
    "選項4": "走路",
    "答案": "2"
  }
]

```

## 6️⃣ Training Configuration

| Parameter | Description | Value |
|------------|-------------|--------|
| Quantization | Type | 4-bit (NF4) |
| LoRA | r / α | 32 / 64 |
| Batch Size | Per device | 1 |
| Gradient Accumulation | Steps | 8 |
| Learning Rate | — | 4e-5 |
| Max Sequence Length | — | 2880 |
| Hardware | GPUs | 5 × NVIDIA RTX 4090 |


## 7️⃣ Result and Conclusion

- Training and evaluation logs are stored in `output_result/` and `gemma3_12b_cpt_sft.log`.

- Detailed inference results (accuracy, predictions, and error cases) are saved in result.txt or the JSON file specified with `--output_file`.

- Final models are located in:
    ```
    models/final_gemma3_12b_cpt_<DATE>/
    models/final_gemma3_12b_sft_<DATE>/
    ```
- W&B dashboard provides real-time plots of loss, learning rate, and GPU usage.
    


