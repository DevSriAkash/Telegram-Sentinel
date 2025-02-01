from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
import csv
import os

api_id = 20631676
api_hash = "63bcb39daa31967465771f41776f8903"
session_name = "telegram_scraper"  


client = TelegramClient(session_name, api_id, api_hash)

async def user_login():
    """Logs in as a real Telegram user."""
    await client.start()
    print("✅ Logged in successfully!")

    if not await client.is_user_authorized():
        phone = input("Enter your phone number with country code: ")
        await client.send_code_request(phone)
        try:
            code = input("Enter the OTP you received: ")
            await client.sign_in(phone, code)
        except SessionPasswordNeededError:
            password = input("Enter your Telegram password: ")
            await client.sign_in(password=password)

async def scrape_messages(channel_link, limit=100):
    """Scrapes messages from a Telegram channel."""
    await user_login()  

    print(f"📌 Scraping messages from {channel_link}...")
    messages = []
    async for message in client.iter_messages(channel_link, limit=limit):
        if message.text:
            messages.append([message.date, message.sender_id, message.text])

    
    with open("telegram_scraped_messages.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Sender", "Message"])
        writer.writerows(messages)
    
    print("✅ Messages saved to 'telegram_scraped_messages.csv'")

async def main():
    """Main function to scrape a Telegram channel."""
    channel_link = "https://t.me/chat2miners"  
    await scrape_messages(channel_link, limit=50)


with client:
    client.loop.run_until_complete(main())
