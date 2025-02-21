import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import inferless
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = inferless.Cls(gpu="A10")

class InferlessPythonModel:
    @app.load
    def initialize(self):
        model_id = 'Deci/DeciLM-7B'
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",load_in_4bit=True,trust_remote_code=True)
        self.qtq_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    @app.infer
    def infer(self, inputs):
        prompt = inputs["prompt"]
        out = self.qtq_pipe(prompt, max_new_tokens=256, do_sample=True, top_p=0.9,temperature=0.9)
        generated_text = out[0]["generated_text"][len(prompt):]

        return {'generated_result': generated_text}

    def finalize(self):
        self.qtq_pipe = None
