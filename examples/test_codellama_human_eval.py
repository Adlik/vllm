import os
import torch
import datetime
import argparse
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

device = "cuda"
problems = read_problems()

parser = argparse.ArgumentParser(description="evaluate codellama human eval.")
parser.add_argument("--model-path", type=str,
                    default="/root/shared/models/huggingface/codellama-7b-base-search-4-bits")
parser.add_argument("--output-path", type=str, default=None,
                    help="output path")
parser.add_argument("--round", type=int, default=1,
                    help="The number of repeated tests")
args = parser.parse_args()

num_samples_per_task = 1


def get_current_time():
    """
    get current time
    Returns:

    """
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime("%Y%m%d%H%M")


# Create a sampling params object.
out_new_token_len = 2048
sampling_params = SamplingParams(
    best_of=1,
    n=1,
    max_tokens=out_new_token_len,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    use_beam_search=False,
    ignore_eos=False,
    presence_penalty=0.0,
    frequency_penalty=0.0,
)

# Create an LLM.
model = llm = LLM(model=args.model_path, tokenizer=args.model_path,
                  tokenizer_mode='auto',
                  dtype='float16',
                  seed=42,
                  tensor_parallel_size=1,
                  trust_remote_code=True,
                  gpu_memory_utilization=0.95,
                  auto_quant_mode='weight_int4'
                  )

for round in range(1, args.round+1):
    print(f"第 {round} 轮测试开始：")
    a = 0
    total = []
    for task_id in tqdm(problems):
        print("task_id: ", task_id)
        input = problems[task_id]["prompt"]
        user_input = f"Below is an instruction that describes a task. Write a response that " \
                     f"appropriately completes the request.\n\n\n### Instruction:\nCreate a Python script for this " \
                     f"problem:\n{input}\n\n### Response:"
        print("user_input:\n" + user_input)
        print(task_id, " ================user input size ", len(user_input))

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = model.generate(user_input, sampling_params)

        for output in outputs:
            for i in range(num_samples_per_task):
                re_process1 = output.outputs[i].text
                print(f"re_process1: {re_process1}")
                re_process = re_process1.split("### Response:")[-1]
                if '```python' in re_process:
                    def_line = re_process.index('```python')
                    re_process = re_process[def_line:].strip()
                    re_process = re_process.replace('```python', '')
                    try:
                        next_line = re_process.index('```')
                        re_process = re_process[:next_line].strip()
                    except:
                        a += 1
                        print("re_process:\n" + re_process)
                        print("===========================\n")

                if "__name__ == \'__main__\'" in re_process:
                    next_line = re_process.index("if __name__ == '__main__':")
                    re_process = re_process[:next_line].strip()

                if "# Example usage" in re_process:
                    next_line = re_process.index('# Example usage')
                    re_process = re_process[:next_line].strip()

                if "###" in re_process:
                    next_line = re_process.index("###")
                    re_process = re_process[:next_line].strip()

                if "</s>" in re_process:
                    next_line = re_process.index("</s>")
                    re_process = re_process[:next_line].strip()
                print("re_process_result:\n" + re_process)
                code_gen = dict(task_id=task_id, completion=re_process)
                print(f"generated:\n{code_gen}")
                total.append(code_gen)

        del outputs
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    base_path = f'examples/json_data/codellama/{args.output_path}'
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    filename = os.path.join(base_path, f'codellama-7b-{get_current_time()}.jsonl')
    write_jsonl(filename, total)
    print(f'Save file path: {filename}')

