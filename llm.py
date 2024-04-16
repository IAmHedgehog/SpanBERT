from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    def __init__(self, device="cuda:4"):
        LLM_path = "mistralai/Mistral-7B-Instruct-v0.2"
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(LLM_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_path)

    def chat(self, message):
        messages = [{"role": "user", "content": message}]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs, max_new_tokens=1000,
            do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return self.get_output(decoded[0])

    def get_output(self, response):
        response = response.replace('<s>', '').replace('</s>', '')
        if '[/INST]' not in response:
            return response.replace('[INST]', '')
        reponse = response.split('[/INST]')[1]
        return reponse.strip()
