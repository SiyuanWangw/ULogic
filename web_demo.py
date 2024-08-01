import json
import torch
import streamlit as st
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from utils.prompter import Prompter
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

st.set_page_config(page_title="Inferential Rule Distillation")
st.title("Inferential Rule Distillation Demo 🤖️")
num_beam_groups = 3

@st.cache_resource
def init_model():
    base_model = './lora-alpaca/mistral-7b-instruct'
    model_path = './lora-alpaca/mistral-7b-instruct-d0_lr2e-4_epoch1_bs16_len512_wp0.05_lora16_8_quan_chat'
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True, torch_dtype=torch.float16, device_map='auto')
    model = PeftModel.from_pretrained(
        model,
        model_path,
        torch_dtype=torch.float16,
    )
    print("model loaded", model.device)
    generation_config = GenerationConfig(
            # temperature=1.0,
            # top_p=0.75,
            # top_k=40,
            # num_beams=num_beam_groups*2,
            # num_beam_groups=num_beam_groups,
            # num_return_sequences=num_beam_groups*2,
            # diversity_penalty=0.2,
            num_beams = 3,
            num_return_sequences=1,
            pad_token_id=0
        )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    prompter = Prompter("")
    print("model init")
    return model, tokenizer, generation_config, prompter

def generate_output(model, tokenizer, generation_config, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=100,
        )
    output = []
    for each in generation_output.sequences:
        output.append(tokenizer.decode(each))
    return output


def clear_chat_history():
    del st.session_state.messages


def init_chat_history(model, tokenizer, generation_config, prompter):
    if "visibility" not in st.session_state:
        st.session_state.visibility = True
    if "conc_visibility" not in st.session_state:
        st.session_state.conc_visibility = True
    if "prem_visibility" not in st.session_state:
        st.session_state.prem_visibility = False

    def change_visible():
        if 'instruction' in st.session_state:
            st.session_state.visibility = "please generate its premise" not in st.session_state.instruction
            st.session_state.prem_visibility = "please generate its premise" in st.session_state.instruction
            st.session_state.conc_visibility = "please generate its conclusion" in st.session_state.instruction

    st.markdown("As an Inference Engine, I can conduct the following inferences on Commonsense Rules 💖")
    # st.markdown("""
    #         - Conclusion Generation
    #         - Premise Completion
    #         - Premise Generation
    #         """)
    with st.expander("###### Conclusion Generation"):
        st.write("Enter a premise describing everyday situations, output its conclusion.")
        st.write("**Example:**")
        st.write('''
        > Input:  
        *Premise: If Person X is of Age Z1 and the minimum age to drive Vehicle Y is Age Z2, and Age Z1 is smaller than Age Z2.*
        ''')
        st.write('''  
        > Output:  
        *Conclusion: Person X can not drive Vehicle Y.*
        ''')
    with st.expander("###### Premise Completion"):
        st.write("Enter a conclusion and its partial premise, output the remaining part of the premise.")
        st.write("**Example:**")
        st.write('''
        > Input:  
        *Conclusion: Person X can not drive Vehicle Y.*  
        *Premise: If Person X is of Age Z1 and the minimum age to drive Vehicle Y is Age Z2,*
        ''')
        st.write('''
        > Output:  
        *Age Z1 is smaller than Age Z2.*
        ''')
    with st.expander("###### Premise Generation"):
        st.write("Enter a conclusion, output its plausible premises.")
        st.write("**Example:**")
        st.write('''
        > Input:  
        *Conclusion: Person X can not drive Vehicle Y.*
        ''')
        st.write('''
        > Output:  
        *Premise: If Person X is of Age Z1 and the minimum age to drive Vehicle Y is Age Z2, and Age Z1 is smaller than Age Z2.*
        ''')
    
    st.write('')
    # st.markdown("------")
    st.markdown('### Instruction')
    option = st.selectbox(
    'Please select an inference type',
    ('Given the premise, please generate its conclusion.', 
    'Given the conclusion and a part of its premise, please complete the remaining portion of the premise.', 
    'Given the conclusion, please generate its premise.'),
    key="instruction",
    on_change=change_visible)

    fact_num = st.selectbox(
    'Please select your preferred fact number in the premise',
    ('1', '2', 'more than 2'),
    disabled=st.session_state.visibility)

    st.write('')
    st.markdown('### Input')
    input_premise = st.text_area(
        "Premise:",
        "If Person X is of Age Z1 and the minimum age to drive Vehicle Y is Age Z2, and Age Z1 is smaller than Age Z2.",
        placeholder="If Person X is of Age Z1 and the minimum age to drive Vehicle Y is Age Z2, and Age Z1 is smaller than Age Z2.",
        disabled=st.session_state.prem_visibility
    )
    input_conclusion = st.text_area(
        "Conclusion:",
        "Person X can not drive Vehicle Y.",
        placeholder="Person X can not drive Vehicle Y.",
        disabled=st.session_state.conc_visibility
    )
    
    if st.button("Submit"):
        if option == 'Given the premise, please generate its conclusion.':
            input_text = "Premise: " + input_premise.strip()
            input_text += "." if input_text[-1] != "." else ""
        elif option == 'Given the conclusion, please generate its premise.':
            input_text = "Conclusion: " + input_conclusion.strip()
            input_text += "." if input_text[-1] != "." else ""
            if fact_num != 'None':
                option = option[:-1] + f" with {fact_num} facts."
        else:
            input_text = "Conclusion: " + input_conclusion.strip()
            input_text += "." if input_text[-1] != "." else ""
            input_text += " \nPremise: " + input_premise.strip()
            input_text += "," if input_text[-1] != "," else ""
        prompt = prompter.generate_prompt(option, input=input_text)
        print("*"*20, "Input")
        print(prompt)
        outputs = generate_output(model, tokenizer, generation_config, prompt)
        # print("$"*20, "Output")
        # print(outputs)
        # print("%"*20)
        st.write('')
        st.markdown("------")
        st.markdown('### Output')
        with open("demo_output/output1.txt", "a") as w_f:
            w_f.write("*"*20+"Input"+"\n")
            w_f.write(prompt+"\n")
            w_f.write("$"*20+"Output"+"\n")

            candidates = []
            for i, each_output in enumerate(outputs):
            # for i in range(num_beam_groups):
                # each_output = outputs[i*2]
                output_text = re.split("#END#|#END #|# END#|# END #", prompter.get_response(each_output))[0].strip()
                if i == 0:
                    candidates.append(output_text)
                else:
                    if output_text in candidates:
                        each_output = outputs[i*2+1]
                        output_text = re.split("#END#|#END #|# END#|# END #", prompter.get_response(each_output))[0].strip()
                        candidates.append(output_text)
                st.write(str(i+1)+": "+output_text)
                w_f.write(str(i+1)+": "+output_text+"\n")
                # print(str(i+1)+": "+output_text)
            w_f.write("%"*20+"\n\n")

def main():
    model, tokenizer, generation_config, prompter = init_model()
    init_chat_history(model, tokenizer, generation_config, prompter)

if __name__ == "__main__":
    main()
