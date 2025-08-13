# About this project/repository

This repository is a document repository used to guide developers:

**Use langchain, gradio, Gemini, and LybicSDK 
to build a playground to demonstrate the Lybic 
sandbox capabilities and large language model agent 
development**

In other words: how to build a computer-use Agent.

What is LybicSDK?

```
Lybic is a GUI agent infra service, provides ready-to-use GUI boxes for agent developers.

While building GUI agent, we may need to run a GUI box for our agent tasks, isolated and hosted on the cloud, 
without setting up instances and images myself.
```

## Project structure

- `assets/`: This directory stores screenshots of actual code execution and web page content, which are used to add image description files to `README.md`. You generally do not need to read or write to this folder.
- `steps/`: This directory stores the final code after the document writer completes each step. The document writer manually copies `main.py` into it according to a certain naming rule. So you generally do not need to read or write to this folder.
- `main.py`: The document writer needs to dynamically modify the Python code file when writing the document, and the document writer needs your help to debug the code.
- `README.md`: The document writer will write a certain guide and write it into this file. The document writer needs your help to optimize this document.

## Build an Agent

The architecture of this agent development is actually very simple, executing according to the following workflow:

1. The user enters a command: for example, "Book me a flight from Los Angeles to Atlanta next Tuesday around 3 PM, and a hotel room near the Atlanta airport on Booking.com."
2. The agent then uses the LybicSDK to capture a screenshot from the sandbox.
3. Enters a loop: The screenshot, along with the user's command and System Prompts (and the global context and memory from the previous step), is sent to the LLM, allowing the LLM to make an overall (or next) plan.
4. The LLM generates an overall plan, such as "I'm currently on my desktop. I need to open a browser, go to Booking.com, and select Book..."
5. The overall plan is stored as global context and memory, and is passed as an additional "overall plan" with each request.
6. The plan is broken down and executed: for example, "Open a browser -> Get the browser's desktop coordinates."
7. Generate an action command.
8. Pass the action command to the LybicSDK for execution.
9. Return to the beginning of step 3 and loop until the LLM deems the operation complete, exiting the loop.

## Note

1. Your main job is to help the document writer implement this agent **step by step**.
2. Don't try to do all the work at once, as that will prevent the document writer from documenting every step.
3. After receiving the instructions, you can determine the current progress of the documentation and code writing by reading the last few lines of `README.md`.

## LybicSDK Usage:

LybicSDK is a GUI automation infrastructure service that provides pre-built GUI environments for agent developers.

The client used by this SDK is asynchronous and requires asynchronous development logic.

```python
import asyncio
from lybic import LybicClient, Sandbox

async def main():
    async with LybicClient() as client:
        sandbox = Sandbox(client)
        new_sandbox_1 = await sandbox.create(name="my-sandbox-1")
        print(new_sandbox_1)
        new_sandbox_2 = await sandbox.create(name="my-sandbox-2")
        print(new_sandbox_2)
        
if __name__=='__main__':
    asyncio.run(main())
```
