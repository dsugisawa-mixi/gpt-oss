from vllm import LLMEngine, EngineArgs, SamplingParams, TokensPrompt


class TokenGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
    ):
        args = EngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.6,
            max_model_len=max_model_len,
            max_num_seqs=1,
            enforce_eager=True,
            kv_cache_dtype="fp8",
        )
        self.engine = LLMEngine.from_engine_args(args)
        self.request_id = 0

    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int] | None = None,
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        if max_tokens == 0:
            max_tokens = None
        request_id = str(self.request_id)
        self.request_id += 1
        sampling_params = SamplingParams(temperature=temperature,
                                         max_tokens=max_tokens,
                                         stop_token_ids=stop_tokens,
                                         logprobs=0 if return_logprobs else None)
        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
        self.engine.add_request(request_id, prompt, sampling_params)
        last_token_id = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            if not step_outputs or not step_outputs[0].outputs:
                continue
            output = step_outputs[0].outputs[0]
            token_ids = output.token_ids
            logprobs_list = output.logprobs if hasattr(output, "logprobs") else None
            new_token_ids = token_ids[len(last_token_id):]
            new_logprobs = logprobs_list[len(last_token_id):] if logprobs_list is not None else [None] * len(new_token_ids)
            for token_id, logprobs in zip(new_token_ids, new_logprobs):
                last_token_id.append(token_id)
                if return_logprobs:
                    logprob_val = None
                    if logprobs is not None and token_id in logprobs:
                        logprob_val = logprobs[token_id].logprob
                    yield (token_id, logprob_val)
                else:
                    yield token_id
                if stop_tokens is not None and token_id in stop_tokens:
                    break
