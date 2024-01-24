from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import warnings
warnings.filterwarnings('ignore')


def llama2_prompt(input_text):
  return f'### Instruction:\n{input_text}\n\n### Response:'

def llama2_output(ouput_text):
  sep = ouput_text[0]['generated_text'].split('### Response:')[1].split('### Instruction')[0].split('## Instruction')[0].split('# Instruction')[0].split('Instruction')[0]
  sep = sep[1:] if sep[0] == '.' else sep
  sep = sep[:sep.find('.')+1] if '.' in sep else sep
  return sep

adapter_name = 'pratik33/nlpai-lab_kullm-polyglot-12_8b-v2_custom-35'


compute_dtype = getattr(torch, 'float16')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2', quantization_config=bnb_config, device_map={'': 0})
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained('nlpai-lab/kullm-polyglot-12.8b-v2', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



model = PeftModel.from_pretrained(model, adapter_name)

pipe = pipeline(task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=100,
                do_sample=True,
                temperature=0.1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                top_k=3,
                # top_p=0.3,
                repetition_penalty = 1.3,
                framework='pt'
                # early_stopping=True
)

def get_response(text):
  result = pipe(llama2_prompt(text))
  return result