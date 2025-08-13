import os
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

import gradio as gr

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["history", "user_input"],
    template="You are a helpful assistant. Below is the conversation history:\n{history}\n\nUser input: {user_input}\n\nAssistant response:"
)

memory = ConversationBufferMemory(memory_key="history")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

def playground(user_input, history=None):
    if not user_input or user_input.strip() == "":
        return "It looks like you didn't provide any text for me to respond to! Please type or paste your input, and I'll be happy to help.", history or []
    try:
        response = chain.invoke({"user_input": user_input})["text"]
        new_history = history + [[user_input, response]] if history else [[user_input, response]]
        return "", new_history
    except Exception as e:
        return f"Error: {str(e)}", history

with gr.Blocks() as iface:
    gr.Markdown("# Playground with LangChain & Gradio")
    gr.Markdown("Interact with the Gemini model and maintain conversation history.")

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Enter your question or prompt here...")
    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=playground,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    )

    user_input.submit(
        fn=playground,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    )

if __name__ == "__main__":
    iface.launch()
