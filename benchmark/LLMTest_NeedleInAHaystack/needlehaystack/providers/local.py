import os
from operator import itemgetter
from typing import Optional
import torch

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM

import tiktoken

from .model import ModelProvider


class Local(ModelProvider):
    """
    A wrapper class for interacting with OpenAI's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the OpenAI model to use for evaluations and interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 300,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        Initializes the OpenAI model provider with a specific model.

        Args:
            model_name (str): The name of the OpenAI model to use. Defaults to 'gpt-3.5-turbo-0125'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """
        self.model_or_path = model_name
        self.model_name = model_name.split("/")[-1]
        self.model_kwargs = model_kwargs

        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_or_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
        self.model.gradient_checkpointing_enable()
 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_or_path, trust_remote_code=True)
    
    async def evaluate_model(self, prompt):
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        MAX_GEN_LENGTH = 512
        if self.model_or_path == "sudy-super/C-cubed-8B-128k":
            self.tokenizer.text_tokenizer.pad_token = self.tokenizer.text_tokenizer.eos_token
            self.tokenizer.context_tokenizer.pad_token = self.tokenizer.context_tokenizer.eos_token
            self.tokenizer.text_tokenizer.pad_token_id = self.tokenizer.text_tokenizer.eos_token_id
            self.tokenizer.context_tokenizer.pad_token_id = self.tokenizer.context_tokenizer.eos_token_id
            
            tokenized_prompts = self.tokenizer.context_tokenizer(prompt["context"], return_tensors="pt", add_special_tokens=False, padding=False)
            context_input_ids = tokenized_prompts.input_ids.cuda()
            tokenized_prompts = self.tokenizer.text_tokenizer(prompt["text"], return_tensors="pt", add_special_tokens=False, padding=False)
            input_ids = tokenized_prompts.input_ids.cuda()
        else:
            tokenized_prompts = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=False)
            input_ids = tokenized_prompts.input_ids.cuda()

        if self.model_or_path == "sudy-super/C-cubed-8B-128k":
            generation_output = self.model.generate(
                context_input_ids=context_input_ids,
                input_ids=input_ids,
                max_new_tokens=MAX_GEN_LENGTH,
                # do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=151645, # <|im_end|>
                use_cache=True,
                return_dict_in_generate=True)
            output = self.tokenizer.text_tokenizer.decode(generation_output.sequences[:,input_ids.shape[1]:][0])
        else:
            generation_output = self.model.generate(
                input_ids,
                max_new_tokens=MAX_GEN_LENGTH,
                # do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=151645, # <|im_end|>
                use_cache=True,
                return_dict_in_generate=True)

            output = self.tokenizer.decode(generation_output.sequences[:,input_ids.shape[1]:][0])
        return output
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        query =  f"""<|im_start|>system
You are a helpful AI bot that answers questions for a user. Keep your response short and direct<|im_end|>
<|im_start|>user
{context}

{retrieval_question} Don't give information outside the document or repeat your findings<|im_end|>
<|im_start|>assistant
"""
        query = {"context": context, "text": f"<|im_start|>system\nYou are a helpful AI bot that answers questions for a user. Keep your response short and direct<|im_end|>\n<|im_start|>user\n{retrieval_question} Don't give information outside the document or repeat your findings<|im_end|>\n<|im_start|>assistant\n"}

        return query
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.context_tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.context_tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the OpenAI model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatOpenAI(temperature=0, model=self.model_name)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain