import logging

from vllm import LLMEngine, EngineArgs, SamplingParams, TokensPrompt

logger = logging.getLogger(__name__)


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
            # Prefix-cache reuse corrupts engine state on vLLM 0.18.0 with
            # this fp8/eager/sm_89 combo — manifests as `engine.step()`
            # yielding tokens at ~37k tok/s without real forward passes
            # (caught by the runaway guard in your_professor_server.py).
            enable_prefix_caching=False,
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
        my_finished = False
        stale_warned = False
        stale_skip_count = 0
        while not my_finished:
            step_outputs = self.engine.step()
            # vLLM 0.18 quirk: the engine keeps re-emitting the previous
            # (finished) request's RequestOutput on every step() until our
            # request produces output. Naively reading step_outputs[0] gives
            # hundreds of stale tokens at ~30k tok/s and trips the runaway
            # guard. Filter to our own request_id and ignore everything else.
            my_output = None
            stale_rids_this_step = []
            for so in step_outputs or ():
                so_rid = getattr(so, "request_id", None)
                if so_rid == request_id:
                    my_output = so
                else:
                    stale_rids_this_step.append(so_rid)
            if stale_rids_this_step:
                stale_skip_count += 1
                if not stale_warned:
                    stale_warned = True
                    logger.warning(
                        "TokenGenerator.generate: ignoring stale step_outputs "
                        "my_rid=%r stale_rids=%r (further occurrences in this "
                        "call will be summarized at the end)",
                        request_id, stale_rids_this_step,
                    )
            if my_output is None:
                if not self.engine.has_unfinished_requests():
                    break
                continue
            if not my_output.outputs:
                if my_output.finished:
                    my_finished = True
                continue
            output = my_output.outputs[0]
            token_ids = output.token_ids
            logprobs_list = output.logprobs if hasattr(output, "logprobs") else None
            new_token_ids = token_ids[len(last_token_id):]
            new_logprobs = logprobs_list[len(last_token_id):] if logprobs_list is not None else [None] * len(new_token_ids)
            stop_hit = False
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
                    stop_hit = True
                    break
            if stop_hit or my_output.finished:
                my_finished = True
        if stale_skip_count > 1:
            logger.warning(
                "TokenGenerator.generate: my_rid=%r ignored stale "
                "step_outputs on %d steps total",
                request_id, stale_skip_count,
            )
