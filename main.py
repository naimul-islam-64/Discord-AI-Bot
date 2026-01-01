import discord
from discord import app_commands
from discord.ext import commands
import openai
import os
import asyncio
from dotenv import load_dotenv
from collections import deque
import logging

# --- Setup & Configuration ---
# Load environment variables
load_dotenv()

# Logger setup for professional debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("DiscordLLM")

# Configuration Constants
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("AI_MODEL_NAME", "gpt-5-mini")

# Premium "Human-like" Colors (Deep Slate / Gunmetal - No generic purple gradients)
EMBED_COLOR_SUCCESS = 0x2C3E50 
EMBED_COLOR_ERROR = 0xB03A2E   
EMBED_COLOR_INFO = 0x5D6D7E     

# Initialize OpenAI Client with custom endpoint
client = openai.AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Discord Intents (Required for reading messages)
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

# Bot Instance
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Memory Management ---
# Stores the last 15 messages per channel to maintain context
# Structure: {channel_id: deque([{role, content}, ...], maxlen=15)}
conversation_history = {}

def get_history(channel_id):
    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=15)
    return conversation_history[channel_id]

def add_to_history(channel_id, role, content):
    history = get_history(channel_id)
    history.append({"role": role, "content": content})

def clear_history(channel_id):
    if channel_id in conversation_history:
        conversation_history[channel_id].clear()

# --- Helper Functions ---

def split_message(text, limit=1900):
    """Splits long AI responses into chunks Discord can handle."""
    chunks = []
    while len(text) > limit:
        # Try to split at the last newline before the limit
        split_index = text.rfind('\n', 0, limit)
        if split_index == -1:
            split_index = limit  # No newline, force split
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    return chunks

async def generate_response(history):
    """Calls the custom LLM API."""
    try:
        # System prompt to enforce helpfulness and persona
        system_msg = {
            "role": "system", 
            "content": "You are a helpful, intelligent assistant on a Discord server. Keep responses concise unless asked for detail. Use markdown formatting."
        }
        
        messages = [system_msg] + list(history)
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"API Error: {e}")
        return f"**Error:** I encountered an issue connecting to the neural core ({e})."

# --- Bot Events ---

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    logger.info(f"Connected to Custom API: {BASE_URL} using model: {MODEL_NAME}")
    
    # Sync Slash Commands
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s).")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

    # Set status
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.listening, 
        name=f"conversations | /reset"
    ))

@bot.event
async def on_message(message):
    # Ignore messages from self or other bots
    if message.author.bot:
        return

    # Check if bot is mentioned OR if it's a DM
    is_mentioned = bot.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_reply = (message.reference and message.reference.resolved and 
                message.reference.resolved.author == bot.user)

    if is_mentioned or is_dm or is_reply:
        channel_id = message.channel.id
        user_input = message.content.replace(f'<@{bot.user.id}>', '').strip()

        # Input validation
        if not user_input:
            return

        # 1. Update Memory
        add_to_history(channel_id, "user", user_input)

        # 2. Trigger Typing Indicator
        async with message.channel.typing():
            # 3. Generate AI Response
            ai_text = await generate_response(get_history(channel_id))

        # 4. Update Memory with AI Response
        add_to_history(channel_id, "assistant", ai_text)

        # 5. Send Response (Handling chunks)
        chunks = split_message(ai_text)
        for chunk in chunks:
            try:
                await message.reply(chunk, mention_author=False)
            except Exception as e:
                # Fallback if reply fails (e.g., message deleted)
                await message.channel.send(chunk)

    # Process standard commands (!commands) if any
    await bot.process_commands(message)

# --- Slash Commands ---

@bot.tree.command(name="reset", description="Clears the conversation memory for this channel.")
async def reset(interaction: discord.Interaction):
    """Wipes the context memory, starting a fresh topic."""
    clear_history(interaction.channel_id)
    
    embed = discord.Embed(
        title="Memory Wiped",
        description="I have cleared the conversation context for this channel. We can start fresh.",
        color=EMBED_COLOR_SUCCESS
    )
    embed.set_footer(text=f"Model: {MODEL_NAME}")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Checks the connection to the AI engine.")
async def status(interaction: discord.Interaction):
    """Checks latency and API status."""
    
    # Calculate Latency
    latency = round(bot.latency * 1000)
    
    embed = discord.Embed(
        title="System Status",
        color=EMBED_COLOR_INFO
    )
    embed.add_field(name="Gateway Latency", value=f"`{latency}ms`", inline=True)
    embed.add_field(name="AI Engine", value=f"`{MODEL_NAME}`", inline=True)
    embed.add_field(name="API Endpoint", value=f"`{BASE_URL}`", inline=False)
    
    await interaction.response.send_message(embed=embed)

# --- Execution ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.critical("Discord Token is missing! Please check your .env file.")
    else:
        bot.run(DISCORD_TOKEN)
