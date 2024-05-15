import asyncio

async def fetch_data(url):
    await asyncio.sleep(1)  # Simulate network request
    return f"Data from {url}"

async def process_data(data):
    await asyncio.sleep(1)  # Simulate processing
    return data.upper()

async def main():
    urls = ["url1", "url2", "url3"]
    tasks = [asyncio.create_task(fetch_data(url)) for url in urls]

    for task in asyncio.as_completed(tasks):
        data = await task
        processed = await asyncio.create_task(process_data(data))
        print(processed)

asyncio.run(main())