import json
import pandas as pd

input_file = r"C:\Users\zjz1\jackzjz_project\legal-ai-tutor\data\datasets\AI-ModelScope--DISC-Law-SFT\snapshots\master\DISC-Law-SFT-Pair.csv"
output_file = r"C:\Users\zjz1\jackzjz_project\legal-ai-tutor\data\legal_sft_500.json"

print("正在读取 CSV 数据，请稍候...")
df = pd.read_csv(input_file, nrows=500)

data_list = []

for index, row in df.iterrows():
    try:
        # 匹配真实列名：input 和 output
        input_text = str(row.get("input", "")) if pd.notna(row.get("input")) else ""
        output_text = str(row.get("output", "")) if pd.notna(row.get("output")) else ""
        
        if input_text and output_text:
            data_list.append({
                "instruction": input_text,  # 把原来的 input 作为指令
                "input": "",                # input 留空
                "output": output_text       # 把原来的 output 作为回答
            })
    except Exception as e:
        continue

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"✅ 成功提取 {len(data_list)} 条数据，保存至 {output_file}")