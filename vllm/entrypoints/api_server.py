"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import argparse
import json
import os
import ssl
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = None
    prompt_token_ids = None
    if "prompt" in request_dict:
        prompt = request_dict.pop("prompt")
    elif "prompt_token_ids" in request_dict:
        prompt_token_ids = request_dict.pop("prompt_token_ids")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    service_busy = engine.is_service_busy()
    result_str = "  Busy  " if service_busy else "Not Busy"
    print("---------- -------------- ----------")
    print(f"---------- >> {result_str} << ----------")
    print("---------- -------------- ----------")

    # Service busy response
    async def busy_results() -> AsyncGenerator[bytes, None]:
        serving_idx = "[Busy]" + os.environ.get("VLLM_SERVING_IDX")

        ret = {
            "text": ["模型推理服务繁忙，请稍后再试。"],
            "input_tokens": [],
            "output_tokens": [[]],
            "serving_idx": serving_idx,
        }
        yield (json.dumps(ret) + "\0").encode("utf-8")

        ret_end = {
            "text": ["模型推理服务繁忙，请稍后再试。"],
            "input_tokens": [],
            "output_tokens": [[]],
            "serving_idx": serving_idx,
            "busy": True,
        }
        yield (json.dumps(ret_end) + "\0").encode("utf-8")

    if service_busy:
        if stream:
            return StreamingResponse(busy_results())

        serving_idx = "[Busy]" + os.environ.get("VLLM_SERVING_IDX")
        ret = {
            "text": ["模型推理服务繁忙，请稍后再试。"],
            "input_tokens": [],
            "output_tokens": [[]],
            "serving_idx": serving_idx,
        }
        return JSONResponse(ret)

    assert engine is not None
    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id,
                                        prompt_token_ids)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            ret = {
                "text":
                    [output.text for output in request_output.outputs],
                "input_tokens":
                    request_output.prompt_token_ids,
                "output_tokens":
                    [output.token_ids for output in request_output.outputs],
            }
            serving_idx = os.environ.get("VLLM_SERVING_IDX")
            if serving_idx:
                ret["serving_idx"] = serving_idx
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    ret = {
        "text": [output.text for output in final_output.outputs],
        "input_tokens": final_output.prompt_token_ids,
        "output_tokens": [output.token_ids for output in final_output.outputs],
    }
    serving_idx = os.environ.get("VLLM_SERVING_IDX")
    if serving_idx:
        ret["serving_idx"] = serving_idx
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
