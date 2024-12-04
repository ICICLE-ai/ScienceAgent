class LLMEngine():
    def __init__(self, llm_engine_name, cache_dir, n_gpus):
        self.llm_engine_name = llm_engine_name
        self.engine = None
        if llm_engine_name.startswith("gpt"):
            from engine.openai_engine import OpenaiEngine
            self.engine = OpenaiEngine(llm_engine_name)
        else:
            from engine.bedrock_engine import BedrockEngine
            self.engine = BedrockEngine(llm_engine_name)

            # from engine.vllm_engine import VllmEngine
            # self.engine = VllmEngine(llm_engine_name, cache_dir, n_gpus)

    def respond(self, user_input, temperature, top_p):
        return self.engine.respond(user_input, temperature, top_p)