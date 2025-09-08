import os

import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from lybic import LybicClient, Sandbox

import gradio as gr

SYSTEM_PROMPT="""You are a GUI Agent, proficient in the operation of various commonly used software on Windows, Linux, and other operating systems.
Please complete the user's task based on user input, history Action, and screen shots.
The user will upload a current screenshot and input an instruction.
You need to complete the entire task step by step, and output only one Action at a time, please strictly follow the format below.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c')  # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx')  # Use escape characters \', \", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content, and next action use hotkey(key='enter')
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')  # Show more information on the `direction` side.
wait()  # Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx')  # Use escape characters \', \", and \\n in content part to ensure we can parse the content in normal python string format.
call_user()  # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
save_memory(content='content')  # When the user explicitly says "remember..." or something similar, `save_memory` is automatically called to save the memory. Next action use finished
output(content='content')  # It is only used when the user specifies to use output, and after output is executed, it cannot be executed again.

## Note
- Use English in `Thought` part.
- The x1,x2 and y1,y2 are the coordinates of the element.
- The resolution of the screenshot is 1280*720.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- You MUST ensure that the `Thought` and `Action` you output are on separate lines, rather than using `\\n` to combine the two lines into one."""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="history")

async def playground(user_input, history=None):
    if not user_input or user_input.strip() == "":
        return (
            "It looks like you didn't provide any text for me to respond to! Please type or paste your input, and I'll be happy to help.",
            history or [],
        )
    try:
        async with LybicClient() as client:
            sandbox = Sandbox(client)
            screenshot_url, _, base64_str = await sandbox.get_screenshot(os.getenv("SANDBOX"))

            user_message_for_history = f"{user_input}\n\n![screenshot]({screenshot_url})"
            message = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"This is user's instruction: {user_input}"},
                        {"type": "text", "text": f"This is user's screenshot,and the resolution of the screenshot is 1280*720: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/webp;base64,{base64_str}"},
                        },
                    ]
                ),
            ]

            system_message = SystemMessage(content=SYSTEM_PROMPT)

            history_messages = memory.chat_memory.messages
            full_message_list = [system_message] + history_messages + message

            response_message = llm.invoke(full_message_list)
            response = response_message.content

            memory.chat_memory.add_message(message[0])
            memory.chat_memory.add_ai_message(response)

            new_history = (
                history + [[user_message_for_history, response]] if history else [[user_message_for_history, response]]
            )
            return "", new_history
    except Exception as e:
        return f"Error: {str(e)}", history

with gr.Blocks() as iface:
    gr.Markdown("# Playground with LangChain & Gradio")
    gr.Markdown("Interact with the Gemini model and maintain conversation history.")

    chatbot = gr.Chatbot(render_markdown=True, sanitize_html=False)
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
