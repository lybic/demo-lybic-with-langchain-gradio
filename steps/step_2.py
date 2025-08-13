import os
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from lybic import LybicClient, Sandbox

import gradio as gr

prompt = PromptTemplate(
    input_variables=["history", "user_input"],
    template="You are a helpful assistant. Below is the conversation history:\n{history}\n\nUser input: {user_input}\n\nAssistant response:",
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="history")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

async def playground(user_input, history=None):
    if not user_input or user_input.strip() == "":
        return (
            "It looks like you didn't provide any text for me to respond to! Please type or paste your input, and I'll be happy to help.",
            history or [],
        )
    try:
        async with LybicClient() as client:
            sandbox = Sandbox(client)
            _, _, base64_str = await sandbox.get_screenshot(os.getenv("SANDBOX"))
            message = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_input},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/webp;base64,{base64_str}"},
                        },
                    ]
                ),
            ]

            history_messages = memory.chat_memory.messages
            full_message_list = history_messages + message

            response_message = llm.invoke(full_message_list)
            response = response_message.content

            memory.chat_memory.add_message(message[0])
            memory.chat_memory.add_ai_message(response)

            new_history = (
                history + [[user_input, response]] if history else [[user_input, response]]
            )
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
