import os
import openai
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class Chatbot:
    def __init__(self, model_name, knowledge_base_path):
        self.embeddings = OpenAIEmbeddings()
        self.path_to_root = "../"
        self.knowledge_base_path = os.path.join(self.path_to_root, knowledge_base_path)
        print(self.knowledge_base_path)
        self._load_knowledge_base(self.knowledge_base_path)
        self._load_model(model_name)
        self._load_prompt_template()
        self._load_qa_chain()


    def response(self, query):
        return self.qa_chain({"query": query})["result"]


    def _load_qa_chain(self):
        assert self.LLM, "Model not loaded"
        assert self.vector_store, "Knowledge base not loaded"
        assert self.qa_chain_prompt, "Prompt template not loaded"

        self.qa_chain = RetrievalQA.from_chain_type(
            self.LLM,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_chain_prompt}
            )


    def _load_prompt_template(self):
        template = """You are a helpful assistant, use the following context to answer the question at the end. If you don't know the answer, just say "sorry, I can't answer this, the answer to this question does not appear in my knowledge base", don't try to make up an answer.
        {context}
        Question: {question}
        Helpful Answer:"""
        self.qa_chain_prompt = PromptTemplate(input_variables=["context", "question"],template=template)


    def _load_model(self, model_name):
        self.LLM = ChatOpenAI(model_name=model_name, temperature=0)


    def _load_knowledge_base(self, knowledge_base_path):
        assert self.embeddings, "Embeddings not loaded"
        assert os.path.exists(knowledge_base_path), "Knowledge base not found"

        loader = TextLoader(knowledge_base_path)
        corpus = loader.load()
        text_data = ' '.join([d.page_content for d in corpus])
        # Since the knowledge base is formatted like a markdown, using a markdown header splitter to get splits on headers and header information in metadata.
        split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=split_on)
        md_header_splits = markdown_splitter.split_text(text_data)

        self.vector_store = DocArrayInMemorySearch.from_documents(md_header_splits, self.embeddings)
