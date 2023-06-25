from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
from llama_index import GPTVectorStoreIndex
import torch
from langchain.llms.base import LLM
from transformers import pipeline

class customLLM(LLM):
    # model_name = "google/flan-t5-small"
    model_name = "google/flan-t5-large"
    pipeline = pipeline("text2text-generation", model=model_name, device="cpu", model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


def create_index():
    llm_predictor = LLMPredictor(llm=customLLM())

    hfemb = HuggingFaceEmbeddings()
    embed_model = LangchainEmbedding(hfemb)

    documents = SimpleDirectoryReader('./Test_Data').load_data()

    # set number of output tokens
    num_output = 250
    # set maximun input size
    max_input_size = 512
    # set maximum chunk overlap
    max_chunk_overlap = 0.00001

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

