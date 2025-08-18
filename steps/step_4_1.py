import os
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from lybic import LybicClient, Sandbox, ComputerUse, dto

import gradio as gr

PLANNER_SYSTEM_PROMPT = """You are a GUI Planner Agent, proficient in the operation of various commonly used software on Windows, Linux, and other operating systems.
You need to complete the task based on user input, historical actions by another Grounding Agent(if this is not the first step), and screenshots, and output your planning results as `ActionSummary` and `Action` to guide another Grounding Agent to perform the corresponding action.
The user will upload a current screenshot and input an instruction.
You need to complete the entire task step by step, and output only one Action at a time, please strictly follow the format below.

## Output Format
```
ActionSummary: ...
Action: ...
```

## Action Summary
- Use English in `ActionSummary` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `ActionSummary` part.

## Action Space
click(point='content') # A description of an area click, such as "search box in search area", "OK button in dialog box", "access label in label area"
left_double(point='content') # A description of an area click, such as "search box in search area", "OK button in dialog box", "access label in label area"
right_single(point='content') # A description of an area click, such as "search box in search area", "OK button in dialog box", "access label in label area"
drag(start_point_from='content', to_end_point='content') # A description of an area click, such as "search box in search area", "OK button in dialog box", "access label in label area"
hotkey(key='ctrl c')  # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='content')  # Use escape characters \', \", and \n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \n at the end of content, and next action use hotkey(key='enter')
scroll(point='content', direction='down or up or right or left')  # Show more information on the `direction` side.
wait()  # Sleep for 5s and take a screenshot to check for any changes.
finished(content='content')  # Use escape characters \', \", and \n in content part to ensure we can parse the content in normal python string format.
call_user()  # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
save_memory(content='content')  # When the user explicitly says "remember..." or something similar, `save_memory` is automatically called to save the memory. Next action use finished
output(content='content')  # It is only used when the user specifies to use output, and after output is executed, it cannot be executed again.
"""

GROUNDNING_SYSTEM_PROMPT = """You are a GUI Grounding Agent, proficient in the operation of various commonly used software on Windows, Linux, and other operating systems.
Please complete the GUI Planner Agent's task based on its input, history Action, and screenshots.
The user(GUI Planner Agent) will input a current screenshot and a text including `ActionSummary` and `Action`.
You need to analyze and process `ActionSummary` and `Action` from GUI Planner Agent, and output only one Action at a time, please strictly follow the format below.

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
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \', \", and \n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \n at the end of content, and next action use hotkey(key='enter')
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \', \", and \n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` part.
- The x1,x2 and y1,y2 are the coordinates of the element.
- The resolution of the screenshot is 1280*720.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
"""

sandbox_id = os.getenv("SANDBOX")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
memory = ConversationBufferMemory(memory_key="history")

from langchain_openai import ChatOpenAI
llm_uitars = ChatOpenAI(
    base_url=os.getenv("AEK_MODEL_ENDPOINT"),
    api_key=os.getenv("ARK_API_KEY"),
    model=os.getenv("ARK_MODEL_NAME"),
)

async def playground(user_input, history=None):
    if not user_input or user_input.strip() == "":
        yield (
            "It looks like you didn't provide any text for me to respond to! Please type or paste your input, and I'll be happy to help.",
            history or [],
        )
        return

    if history is None:
        history = []

    try:
        async with LybicClient() as client:
            sandbox = Sandbox(client)
            computer_use = ComputerUse(client)

            # Add the initial user input to the history
            screenshot_url, _, _ = await sandbox.get_screenshot(sandbox_id)
            user_message_for_history = f'{user_input}\n\n![screenshot]({screenshot_url})'
            history.append([user_message_for_history, "Thinking..."])
            yield "", history

            while True:
                # 1. Observe: Get the current screenshot
                screenshot_url, _, base64_str = await sandbox.get_screenshot(sandbox_id)

                # 2. Think: Prepare messages for the LLM
                planner_message = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": f"This is user's instruction: {user_input}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/webp;base64,{base64_str}"},
                            },
                        ]
                    ),
                ]

                system_message = SystemMessage(content=PLANNER_SYSTEM_PROMPT)
                history_messages = memory.chat_memory.messages
                full_message_list = [system_message] + history_messages + planner_message

                # Get LLM response
                response_message = await llm.ainvoke(full_message_list)
                response = response_message.content
                print("Planner output:", response)
                print("---------")

                # Update history with the LLM's thought process
                history[-1][1] = response  # Update the "Thinking..." message
                yield "", history

                # 3. Grounding
                grounding_message = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": response},
                            {"type": "text",
                             "text": f"This is user's screenshot,and the resolution of the screenshot is 1280*720:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/webp;base64,{base64_str}"},
                            }
                        ]
                    )
                ]

                system_message = SystemMessage(content=GROUNDNING_SYSTEM_PROMPT)
                full_message_list = [system_message] + grounding_message
                response_message = await llm_uitars.ainvoke(full_message_list)
                response = response_message.content
                print("Grounding output:", response)

                # 4. Act: Parse and execute the action
                action_dto = await computer_use.parse_model_output(
                    dto.ComputerUseParseRequestDto(model="ui-tars",
                                                   textContent=str(response),
                                                   ))

                action = action_dto.actions[0]
                await computer_use.execute_computer_use_action(sandbox_id, dto.ComputerUseActionDto(action=action))

                # Memory: Store the conversation
                memory.chat_memory.add_message(planner_message[0])
                memory.chat_memory.add_ai_message(response)

                # Check for termination condition
                if isinstance(action, dto.FinishedAction):
                    final_screenshot_url, _, _ = await sandbox.get_screenshot(sandbox_id)
                    history.append([None, f"Task finished. Final state:\n\n![screenshot]({final_screenshot_url})"])
                    yield "", history
                    break

                # Prepare for the next loop iteration
                history.append([f"Action executed: `{ action.model_dump_json() }`\n\nTaking new screenshot...", "Thinking..."])
                yield "", history
    except Exception as e:
        import traceback
        yield f"Error: {str(e)}\n{traceback.format_exc()}", history

with gr.Blocks() as iface:
    gr.Markdown("# Playground with LangChain & Gradio")
    gr.Markdown("Interact with the Gemini model and maintain conversation history.")

    chatbot = gr.Chatbot(render_markdown=True, sanitize_html=False)
    user_input = gr.Textbox(placeholder="Enter your task for the agent...")
    submit_button = gr.Button("Run Agent")

    # Use a generator for the click event
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
