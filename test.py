import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # ✅ 導入 PEFT 來載入 LoRA adapter
from tqdm import tqdm
import re
import argparse
import gc

# 清理 GPU 記憶體的輔助函數
def clear_gpu_memory():
    """清理 GPU 記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_test_data(test_file):
    """載入測試集"""
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_test_prompt(example):
    """格式化測試問題為訓練時的格式"""
    # 構建選項文本
    options_text = f"(1) {example['選項1']}\n(2) {example['選項2']}\n(3) {example['選項3']}\n(4) {example['選項4']}"
    
    # 構建完整輸入
    input_text = f"{example['題目']}\n{options_text}"
    
    # 使用訓練時的 prompt 格式
    user_content = f"Please select the correct answer.\n\n{input_text}"
    # user_content = f"Select a correct answer based on the question and the text, and only output the question number.\n\n{input_text}"

    return user_content

def extract_answer(response):
    """從模型輸出中提取答案"""
    # 嘗試多種模式匹配答案
    patterns = [
        r'答案[是為]?\s*[：:]\s*(\d)',  # 答案是：1 或 答案：1
        r'選項\s*(\d)',  # 選項1
        r'^\s*(\d)\s*$',  # 單獨的數字
        r'\((\d)\)',  # (1)
        r'正確答案[是為]?\s*[：:]\s*(\d)',  # 正確答案是：1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # 如果都找不到，返回第一個出現的數字
    numbers = re.findall(r'\d', response)
    if numbers:
        # 過濾出 1-4 的數字
        valid_numbers = [n for n in numbers if n in ['1', '2', '3', '4']]
        if valid_numbers:
            return valid_numbers[0]
    
    return None

def evaluate_model(model_path, test_file, output_file=None, max_samples=None):
    """評估模型在測試集上的表現"""
    
    print("=" * 60)
    print("🧪 開始評估模型")
    print("=" * 60)
    
    # 先清理 GPU 記憶體
    print("\n🧹 清理 GPU 記憶體...")
    clear_gpu_memory()
    
    # 載入模型和 tokenizer
    print(f"\n📂 載入模型: {model_path}")
    
    # 檢查是否為 LoRA adapter (檢查是否有 adapter_config.json)
    import os
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_lora:
        print("📌 檢測到 LoRA adapter,正在載入基礎模型...")
        # 基礎模型路徑 (Gemma-3-1B)
        base_model_path = "google/gemma-3-12b-it"
        
        # 先清理 GPU 記憶體
        torch.cuda.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,  # 修正: 使用 dtype 而不是 torch_dtype
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True  # 減少 CPU 記憶體使用
        )
        
        print(f"📌 載入 LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # 合併 LoRA 權重到基礎模型
        print("✅ LoRA 模型合併完成")
    else:
        print("📌 載入完整模型...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        # 先清理 GPU 記憶體
        torch.cuda.empty_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,  # 修正: 使用 dtype 而不是 torch_dtype
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True  # 減少 CPU 記憶體使用
        )
    
    model.eval()
    print("✅ 模型載入完成")
    
    # 載入測試集
    print(f"\n📂 載入測試集: {test_file}")
    test_data = load_test_data(test_file)
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"✅ 測試集載入完成，共 {len(test_data)} 題")
    
    # 開始評估
    correct = 0
    total = 0
    results = []
    
    print("\n🔄 開始推理...")
    for idx, example in enumerate(tqdm(test_data, desc="評估進度")):
        # 格式化問題
        user_content = format_test_prompt(example)
        
        # 構建對話格式
        conversation = [
            {"role": "user", "content": user_content}
        ]
        
        # 應用 chat template
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Debug: 顯示第一個問題的 prompt
        if idx == 0:
            print(f"\n{'='*60}")
            print("📝 第一題的 Prompt 範例:")
            print(f"{'='*60}")
            print(prompt[:500])  # 只顯示前 500 字元
            print(f"{'='*60}\n")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # 降低生成長度加快測試
                do_sample=False,  # 使用採樣提高多樣性
                # temperature=1.0,  # 適中的溫度
                # top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解碼輸出
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Debug: 顯示第一個回答 (在清理變量之前)
        if idx == 0:
            generated_tokens = len(outputs[0]) - inputs['input_ids'].shape[1]
            print(f"\n{'='*60}")
            print("📝 第一題的完整輸出:")
            print(f"{'='*60}")
            print(f"生成 tokens 數: {generated_tokens}")
            print(f"回應內容: [{response}]")
            print(f"回應長度: {len(response)} 字元")
            print(f"{'='*60}\n")
        
        # 清理中間變量
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 提取預測答案
        predicted = extract_answer(response)
        ground_truth = str(example['答案'])
        
        # 判斷是否正確
        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1
        total += 1
        
        # 記錄結果
        result = {
            "題號": example['題號'],
            "題目": example['題目'],
            "正確答案": ground_truth,
            "預測答案": predicted,
            "模型回應": response,
            "是否正確": is_correct
        }
        results.append(result)
        
        # 每 10 題顯示一次中間結果
        if (idx + 1) % 10 == 0:
            current_acc = correct / total * 100
            print(f"\n📊 中間結果 ({idx + 1}/{len(test_data)}): 準確率 = {current_acc:.2f}%")
    
    # 計算最終準確率
    accuracy = correct / total * 100
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("📊 評估結果")
    print("=" * 60)
    print(f"總題數: {total}")
    print(f"答對題數: {correct}")
    print(f"答錯題數: {total - correct}")
    print(f"準確率: {accuracy:.2f}%")
    print("=" * 60)
    
    # 儲存詳細結果
    if output_file:
        output_data = {
            "模型路徑": model_path,
            "測試集路徑": test_file,
            "總題數": total,
            "答對題數": correct,
            "答錯題數": total - correct,
            "準確率": f"{accuracy:.2f}%",
            "詳細結果": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 詳細結果已儲存至: {output_file}")
    
    # 顯示一些錯誤案例
    print("\n❌ 錯誤案例範例 (最多顯示 5 個):")
    error_cases = [r for r in results if not r['是否正確']][:5]
    for i, case in enumerate(error_cases, 1):
        print(f"\n--- 錯誤案例 {i} ---")
        print(f"題號: {case['題號']}")
        print(f"題目: {case['題目'][:100]}...")
        print(f"正確答案: {case['正確答案']}")
        print(f"預測答案: {case['預測答案']}")
        print(f"模型回應: {case['模型回應'][:200]}...")
    
    # 清理模型和記憶體
    print("\n🧹 清理記憶體...")
    del model
    del tokenizer
    clear_gpu_memory()
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='評估模型在測試集上的表現')
    parser.add_argument('--model_path', type=str, required=True, help='模型路徑')
    parser.add_argument('--test_file', type=str, required=True, help='測試集 JSON 檔案路徑')
    parser.add_argument('--output_file', type=str, default=None, help='輸出結果 JSON 檔案路徑')
    parser.add_argument('--max_samples', type=int, default=None, help='最大測試樣本數（用於快速測試）')
    
    args = parser.parse_args()
    
    try:
        evaluate_model(
            model_path=args.model_path,
            test_file=args.test_file,
            output_file=args.output_file,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"\n❌ 評估過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        # 確保清理記憶體
        clear_gpu_memory()
        exit(1)