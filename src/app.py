import gradio as gr
import random
import time
import os
import openai
from dotenv import load_dotenv, find_dotenv

from chatbot import Chatbot
from utils.models_and_path import KNOWLEDGE_BASE_PATH, MODEL_NAME


load_dotenv(find_dotenv())
openai.api_key  = os.environ['OPENAI_API_KEY']

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    chat_AI = Chatbot(model_name=MODEL_NAME, knowledge_base_path=KNOWLEDGE_BASE_PATH)

    def respond(message, chat_history):
        generated_output = chat_AI.response(message)
        chat_history.append((message, generated_output))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(share=True)

# To run this demo, run the following command in your terminal:
    # gradio demo chatbot.py
