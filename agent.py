from dotenv import load_dotenv
import os
import io
import requests
import asyncio
import discord

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from tavily import TavilyClient

load_dotenv()


# ============================================================
# API KEYS
# ============================================================

GOOGLE_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)

if not GOOGLE_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY or GOOGLE_API_KEY is missing from .env"
    )


# ============================================================
# TAVILY CLIENT
# ============================================================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ============================================================
# INTERNET SEARCH TOOL
# ============================================================

@tool
def surfInterNet(query: str):
    """
    Use this tool to search the internet and get
    the latest information.
    """

    result = tavily_client.search(
        query=query
    )

    return str(result)


# ============================================================
# IMAGE GENERATION TOOL
# ============================================================

@tool
def generateAndSendImage(
    prompt: str,
    runTime: ToolRuntime
):
    """
    Generate an image using Pollinations AI
    and send it directly to Discord.
    """

    config = runTime.config.get(
        "configurable",
        {}
    )

    message = config.get("message")
    loop = config.get("loop")

    if message is None:
        return "Discord message is missing."

    if loop is None:
        return "Discord event loop is missing."

    try:

        # Get Pollinations API key
        api_key = os.getenv(
            "POLLINATIONS_API_KEY"
        )

        if not api_key:
            return (
                "POLLINATIONS_API_KEY is missing "
                "from .env"
            )

        # Pollinations image API
        url = (
            "https://gen.pollinations.ai/image/"
            + requests.utils.quote(prompt)
        )

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        params = {
            "model": "flux"
        }

        print("Generating image...")
        print("Prompt:", prompt)

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=120
        )

        print(
            "Pollinations status:",
            response.status_code
        )

        if response.status_code != 200:

            print("Pollinations error:")
            print(response.text)

            return (
                f"Image generation failed: "
                f"{response.status_code}"
            )

        # Convert response to image
        image_buffer = io.BytesIO(
            response.content
        )

        file = discord.File(
            fp=image_buffer,
            filename="generated_image.png"
        )

        # Send image to Discord
        future = asyncio.run_coroutine_threadsafe(
            message.channel.send(file=file),
            loop
        )

        future.result()

        print("Image sent successfully.")

        return (
            "Image generated and sent successfully."
        )

    except Exception as e:

        print(
            "IMAGE GENERATION ERROR:",
            repr(e)
        )

        return (
            f"Image generation failed: {str(e)}"
        )


# ============================================================
# GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# ============================================================
# CREATE AGENT
# ============================================================

agent = create_agent(
    model=model,

    tools=[
        surfInterNet,
        generateAndSendImage
    ],

    system_prompt="""
You are a helpful Discord AI assistant.

Provide clean and useful answers.

Use surfInterNet when the user asks for:
- current information
- latest information
- recent news
- web searches
- information that may have changed

Use generateAndSendImage whenever the user asks to:
- generate an image
- create an image
- make a picture
- draw something
- create artwork
- visualize something

When the user asks for an image, ALWAYS use
generateAndSendImage.

Do not tell the user that you cannot generate images.

After the image tool successfully sends the image,
give a short confirmation.
"""
)