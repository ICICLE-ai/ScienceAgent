from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class VllmEngine():

    def __init__(self, llm_engine_name, cache_dir, n_gpus):
        self.llm_engine_name = llm_engine_name
        self.llm = LLM(
            model=llm_engine_name, 
            download_dir=cache_dir, 
            tensor_parallel_size=n_gpus
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_engine_name)

    def respond(self, user_input, temperature):
        sampling_params = SamplingParams(
            temperature=temperature,
            n=1,
            max_tokens=2000
        )

        prompt = self.tokenizer.apply_chat_template(user_input, tokenize=False)

        outputs = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        responses = [o.text for o in outputs[0].outputs]

        return responses[0]