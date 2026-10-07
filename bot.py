from dotenv import load_dotenv
import os

import asyncio
import discord


from langchain_core.messages import HumanMessage
from agent import agent 
load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_message(message):

    # Ignore bot messages
    if message.author.bot:
        return
    async with message.channel.typing():
         content = message.content

    # Invoke agent
    response =  await agent.ainvoke({
        "messages": [HumanMessage(content=content)]},
        config={"configurable": {"message":message,"loop":asyncio.get_running_loop()}}
    )

    # Get AI response
    agent_message = response["messages"][-1].text

    # Send response
    await message.channel.send(agent_message)

client.run(os.getenv("DISCORD_API_KEY"))


