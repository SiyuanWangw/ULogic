import os
import sys
import time
import re

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from utils.prompter import Prompter
import evaluate

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass



def main(
    load_8bit: bool = False,
    base_model: str = "",
    data_path: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        print("lora_weights path**********************", lora_weights)
        model.print_trainable_parameters()
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # case_conc_generation = "### Input:\nPremise: If Person X is banned from Platform Y.\n\n### Response:\nConclusion: Person X cannot communicate on Platform Y. #END#" + \
    #                         "\n\n### Input:\nPremise: If Person X has Tool Z1 and Tool Z1 is suitable for measuring Natural Phenomenon Y.\n\n### Response:\nConclusion: Person X can study Natural Phenomenon Y. #END#"
    # case_prem_completion = "### Input:\nConclusion: Person X can use Furniture Y. \nPremise: If Person X is not sensitive to Material Z,\n\n### Response:\nFurniture Y is made of Material Z. #END#" + \
    #                         "\n\n### Input:\nConclusion: Person X can donate to Organization Y. \nPremise: If Person X works at Job C which pays Money Z1, and the Money Z1 is bigger than Money Z2,\n\n### Response:\nOrganization Y organizes Event D which collects Money Z2. #END#"
    # case_prem_generation = "### Input:\nConclusion: Organization X can access Event Y.\n\n### Response:\nPremise: If Organization X sponsors Event Y. #END#" + \
    #                         "\n\n### Input:\nConclusion: Animal X is at risk of Disease Y.\n\n### Response:\nPremise: If Animal X is adapted to Region Z and Disease Y is endemic in Region Z. #END#"
    num_beam_groups = 3
    def evaluate_process(
        instruction,
        input=None,
        temperature=1.0,
        top_p=0.75,
        top_k=40,
        num_beams=3,
        # num_beams=6,
        max_new_tokens=100,
        stream_output=False,
        **kwargs,
    ):  
        # if instruction == "Please generate the conclusion of the given premise.":
        #     prompt = prompter.generate_prompt(instruction, case=case_conc_generation, input=input)
        # elif instruction == "Please complete the premise of the given conclusion.":
        #     prompt = prompter.generate_prompt(instruction, case=case_prem_completion, input=input)
        # else:
        #     prompt = prompter.generate_prompt(instruction, case=case_prem_generation, input=input)
        # prompt = prompter.generate_prompt(instruction, case=True, input=input)
        prompt = prompter.generate_prompt(instruction, input=input)
        # print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # num_beams=num_beams,
            # num_return_sequences=1,
            # # num_return_sequences=3,
            pad_token_id=0,
            num_beams=num_beams*2,
            num_beam_groups=num_beams,
            num_return_sequences=num_beams*2,
            diversity_penalty=0.5,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        generate_start = time.time()

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        generate_end = time.time()

        # s = generation_output.sequences[0]
        # output = tokenizer.decode(s)
        # return prompter.get_response(output)
        
        output = []
        for i, each in enumerate(generation_output.sequences):
            # if i % 2 == 0:
            output.append(prompter.get_response(tokenizer.decode(each)))
        return output
        

    data = load_dataset("json", data_files=data_path)
    # train_val = data["train"].train_test_split(
    #     test_size=2000, shuffle=True, seed=42
    # )
    # val_data = train_val["test"]
    # val_data = data['train'].shuffle(seed=42)
    val_data = data['train']
    print(type(val_data), len(val_data))

    predictions = []
    # references = []

    from tqdm import tqdm
    # for i in tqdm(range(min(len(val_data), 10))):
    for i in tqdm(range(len(val_data))):
        output = evaluate_process(instruction=val_data[i]['instruction'], input = val_data[i]['input'])
        # output = re.split("#END#|#END #|# END#|# END #", output)[0].strip()

        output = [re.split("#END#|#END #|# END#|# END #", each)[0].strip() for each in output]
        candidates = []
        for j in range(num_beam_groups):
            each_output = output[j*2]
            if j == 0:
                candidates.append(each_output)
            else:
                if each_output in candidates:
                    each_output = output[j*2+1]
                candidates.append(each_output)

        reference = val_data[i]['output'].split("#END#")[0].strip()
        # predictions.append(output)
        predictions.append(candidates)

        if i < 20:
            print("*"*50)
            print(val_data[i]['instruction'])
            print(val_data[i]['input'])
            print("reference:", val_data[i]['output'])
            print("#"*20)
            # print("generated output:", output)
            print("generated output:")
            for each in candidates:
                print(each)

    # with open(lora_weights+"/prem_gen_predictions.txt", "w") as w_f_1:
    # # with open("./lora-alpaca/llama-2/predictions_2demo.txt", "w") as w_f_1:
    #     for i, each in tqdm(enumerate(predictions)):
    #         w_f_1.write(str(i) + ": " + str(each)+"\n")
    #         # w_f_1.write(str(i) + ": \n" + val_data[i]['input'] + "\n" + each+"\n")


if __name__ == "__main__":
    fire.Fire(main)
