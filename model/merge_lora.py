from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys


model_name_or_path = sys.argv[1]
adapter_name_or_path = sys.argv[2]
save_path = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
print("load model success")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")

# python model/merge_lora.py 'model/models/Llama-2-7b-hf' 'model/output/model_base' 'model/models/model_base'
