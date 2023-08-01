import discord
from discord.ext import commands
import sys,os
from discord.message import Message

# import util from parent directory
sys.path.insert(0, os.path.abspath('..'))

from utils import *
from discordbot.commands import *
TOKEN = get_token()


class HomeHub(commands.Bot):
    def __init__(self, command_prefix):
        intents = discord.Intents.default()
        intents.members = True
        super().__init__(command_prefix, intents=intents)
        
    async def on_ready(self):
        print(f"Bot connected as {self.user}")
        
    async def on_message(self, message):
        if message.author.bot:
            return
        print(message.content)
        msg = process_msg(message.content)
        await message.channel.send(msg)

    
    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("Invalid command. Please try again.")
    
    def run_bot(self):
        self.run(TOKEN)
