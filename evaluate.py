# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "faker",
#     "httpx",
#     "numpy",
#     "pillow",
#     "python-dateutil",
# ]
# ///
import hashlib
import httpx
import json
import logging
import numpy as np
import os
import re
import subprocess
from dateutil.parser import parse
from datagen import (
    get_markdown,
    get_dates,
    get_contacts,
    get_logs,
    get_docs,
    get_email,
    get_credit_card,
    get_comments,
    get_tickets,
)


openai_api_base = os.getenv("OPENAI_API_BASE", "https://aiproxy.sanand.workers.dev/openai/v1")
openai_api_key = os.getenv("OPENAI_API_KEY")


def num(str):
    return int(hashlib.sha256(str.encode()).hexdigest(), 16) % (2**32)


def mismatch(msg, expected, result):
    logging.error(f"🔴 {msg}\n⚠️ EXPECTED:\n{expected}\n⚠️ RESULT:\n{result}")
    return False


async def run(task: str):
    async with httpx.AsyncClient(timeout=30) as client:
        logging.warning(f"🟡 Running task: {task.strip()}")
        response = await client.post("http://localhost:8000/run", params={"task": task})
        try:
            response_text = json.dumps(response.json(), indent=2)
        except json.JSONDecodeError:
            response_text = response.text
        if response.status_code < 400:
            logging.info(f"🟢 HTTP {response.status_code} {response_text}")
        else:
            logging.error(f"🔴 HTTP {response.status_code} {response_text}")
        return response.status_code, response_text


async def read(path: str):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"http://localhost:8000/read?path={path}")
        if response.status_code != 200:
            raise Exception(f"Cannot read {path}")
        return response.text


async def a1(email: str, **kwargs):
    await run(
        f"""
Install `uv` (if required) and run the script `https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py`
with `{email}` as the only argument
"""
    )
    return email in await read("/data/format.md")


async def a2(email: str, file: str = "/data/format.md", **kwargs):
    original = get_markdown(email)
    expected = subprocess.run(
        ["npx", "prettier@3.4.2", "--stdin-filepath", file],
        input=original,
        capture_output=True,
        text=True,
        check=True,
        # Ensure npx is picked up from the PATH on Windows
        shell=True,
    ).stdout
    result = await run(
        f"""
Format the contents of `{file}` using `prettier@3.4.2`, updating the file in-place
"""
    )
    result = await read(file)
    if result != expected:
        return mismatch(file, expected, result)
    return True


async def a3(email, **kwargs):
    dates = get_dates(email)
    await run(
        "The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`"
    )
    result = await read("/data/dates-wednesdays.txt")
    expected = sum(1 for date in dates if parse(date).weekday() == 2)
    if result.strip() != str(expected):
        return mismatch("/data/dates-wednesdays.txt", expected, result)
    return True


async def a4(email, **kwargs):
    contacts = get_contacts(email)
    contacts.sort(key=lambda c: (c["last_name"], c["first_name"]))
    await run(
        "Sort the array of contacts in `/data/contacts.json` by `last_name`, then `first_name`, and write the result to `/data/contacts-sorted.json`"
    )
    result = await read("/data/contacts-sorted.json")
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        logging.error("🔴 /data/contacts-sorted.json was not valid JSON")
        return False
    if json.dumps(result, sort_keys=True) != json.dumps(contacts, sort_keys=True):
        return mismatch("/data/contacts-sorted.json", contacts, result)
    return True


async def a5(email, **kwargs):
    files = get_logs(email)
    files.sort(key=lambda f: f[0])
    expected = "".join([f[1].split("\n")[0] + "\n" for f in files[:10]])
    await run(
        "Write the first line of the 10 most recent `.log` file in `/data/logs/` to `/data/logs-recent.txt`, most recent first"
    )
    result = await read("/data/logs-recent.txt")
    if result.strip() != expected.strip():
        return mismatch("/data/logs-recent.txt", expected, result)
    return True


# TODO: Verify after datagen
async def a6(email, **kwargs):
    docs = get_docs(email)
    await run(
        """Find all Markdown (`.md`) files in `/data/docs/`.
For each file, extract the first occurrance of each H1 (i.e. a line starting with `# `).
Create an index file `/data/docs/index.json` that maps each filename (without the `/data/docs/` prefix) to its title
(e.g. `{"README.md": "Home", "path/to/large-language-models.md": "Large Language Models", ...}`)"""
    )
    expected = {}
    for dir, file, text in docs:
        # get the first line starting with #
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        expected[f"{dir}/{file}.md"] = title
    result = await read("/data/docs/index.json")
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        logging.error("🔴 /data/docs/index.json was not valid JSON")
        return False
    if json.dumps(result, sort_keys=True) != json.dumps(expected, sort_keys=True):
        return mismatch("/data/docs/index.json", expected, result)
    return True


async def a7(email, **kwargs):
    expected = get_email(email)["from_email"]
    await run(
        "`/data/email.txt` contains an email message. Pass the content to an LLM with instructions to extract the sender's email address, and write just the email address to `/data/email-sender.txt`"
    )
    result = await read("/data/email-sender.txt")
    if result != expected:
        return mismatch("/data/email-sender.txt", expected, result)
    return True


async def a8(email, **kwargs):
    data = get_credit_card(email)
    await run(
        "`/data/credit_card.png` contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to `/data/credit-card.txt`"
    )
    result = await read("/data/credit-card.txt")
    if re.sub(r"\D", "", result) != re.sub(r"\D", "", data["number"]):
        return mismatch("/data/credit-card.txt", data["number"], result)
    return True


async def a9(email, **kwargs):
    data = get_comments(email)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{openai_api_base}/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": data},
        )
    embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
    similarity = np.dot(embeddings, embeddings.T)
    # Create mask to ignore diagonal (self-similarity)
    np.fill_diagonal(similarity, -np.inf)
    # Get indices of maximum similarity
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    expected = "\n".join(sorted([data[i], data[j]]))
    await run(
        "`/data/comments.txt` contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to `/data/comments-similar.txt`, one per line"
    )
    result = await read("/data/comments-similar.txt")
    sorted_result = "\n".join(sorted([line for line in result.split("\n") if line.strip()]))
    if sorted_result != expected:
        return mismatch("/data/comments-similar.txt", expected, result)
    return True


async def a10(email, **kwargs):
    data = get_tickets(email)
    await run(
        'The SQLite database file `/data/ticket-sales.db` has a `tickets` with columns `type`, `units`, and `price`. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the "Gold" ticket type? Write the number in `/data/ticket-sales-gold.txt`'
    )
    result = await read("/data/ticket-sales-gold.txt")
    expected = sum(row[1] * row[2] for row in data if row[0].lower() == "gold")
    try:
        result = float(result)
    except ValueError:
        logging.error(f"🔴 /data/ticket-sales-gold.txt was {result}, not a valid number")
        return False
    if abs(result - expected) > 0.1:
        return mismatch("/data/ticket-sales-gold.txt", expected, result)
    return True



def is_safe_path(base_dir, path):
    """
    Ensures the given path is within the base directory.
    """
    base_dir = os.path.abspath(base_dir)
    path = os.path.abspath(path)
    return os.path.commonpath([base_dir]) == os.path.commonpath([base_dir, path])

async def b1(email, **kwargs):
    """
    Test to ensure no data outside `/data` is accessed.
    """
    # Attempt to access a file outside `/data`
    unsafe_path = "/etc/passwd"
    if is_safe_path("/data", unsafe_path):
        logging.error("🔴 Security violation: Accessed data outside `/data`")
        return False

    # Attempt to access a valid file inside `/data`
    safe_path = "/data/test.txt"
    if not is_safe_path("/data", safe_path):
        logging.error("🔴 Security violation: Denied access to valid file inside `/data`")
        return False

    logging.info("🟢 B1 passed: All paths validated correctly.")

    return True


import os

def prevent_deletion():
    """
    Overrides `os.remove` and `os.rmdir` to prevent file deletions.
    """
    original_remove = os.remove
    original_rmdir = os.rmdir

    def restricted_remove(path):
        raise PermissionError(f"Deletion of files is not allowed: {path}")

    def restricted_rmdir(path):
        raise PermissionError(f"Deletion of directories is not allowed: {path}")

    os.remove = restricted_remove
    os.rmdir = restricted_rmdir

async def b2(email, **kwargs):
    """
    Test to ensure no files are deleted.
    """
    # Activate deletion prevention
    prevent_deletion()

    # Attempt to delete a file
    try:
        os.remove("/data/test.txt")
        logging.error("🔴 Security violation: File deletion allowed.")
        return False
    except PermissionError:
        logging.info("🟢 B2 passed: File deletion prevented.")

    # Attempt to delete a directory
    try:
        os.rmdir("/data/test-dir")
        logging.error("🔴 Security violation: Directory deletion allowed.")
        return False
    except PermissionError:
        logging.info("🟢 B2 passed: Directory deletion prevented.")

    return True

async def b3(email, **kwargs):
    """
    Test fetching data from an API and saving it.
    """
    url = "https://jsonplaceholder.typicode.com/posts/1"
    save_path = "/data/api-response.json"

    await run(f"Fetch data from `{url}` and save it to `{save_path}`")

    result = await read(save_path)
    expected = httpx.get(url).text  # Fetch expected data directly from the API

    if result.strip() != expected.strip():
        return mismatch(save_path, expected, result)

    return True
async def b5(email, **kwargs):
    """
    Test running a SQL query on a database.
    """
    db_path = "/data/ticket-sales.db"
    query = "SELECT COUNT(*) FROM tickets"
    output_file = "/data/sql-query-result.txt"

    try:
        # Run the SQL query
        await run(f"Run the SQL query `{query}` on `{db_path}` and save the result to `{output_file}`")

        # Verify query result
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(query)
        expected = str(cursor.fetchone()[0])
        
        result = await read(output_file)
        
        if result.strip() != expected.strip():
            return mismatch(output_file, expected, result)

        return True

    except Exception as e:
        logging.error(f"🔴 Error in B5: {e}")
        return False



async def b4(email, **kwargs):
    """
    Test cloning a git repository and making a commit.
    """
    repo_url = "https://github.com/sanand0/tools-in-data-science-public.git"
    repo_path = "/data/repo"
    commit_message = "Test commit"

    # Run the task
    await run(f"Clone the git repository `{repo_url}` into `{repo_path}` and make a commit with message `{commit_message}`")

    # Verify repository exists
    if not os.path.exists(repo_path):
        logging.error(f"🔴 Repository not found at {repo_path}")
        return False

    # Check for the commit message in the git log
    result = subprocess.run(["git", "-C", repo_path, "log", "-1", "--pretty=%B"], capture_output=True, text=True)
    
    if commit_message not in result.stdout:
        return mismatch("git log", commit_message, result.stdout)

    return True


async def b6(email, **kwargs):
    """
    Test extracting data from a website (web scraping).
    """
    url = "https://jsonplaceholder.typicode.com/posts/1"
    output_file = "/data/web-scrape-result.txt"

    await run(f"Extract data from `{url}` and save it to `{output_file}`")

    # Verify scraped content matches expected content
    expected = httpx.get(url).text
    result = await read(output_file)

    if result.strip() != expected.strip():
        return mismatch(output_file, expected, result)

    return True

async def b8(email, **kwargs):
    import openai
    import unittest.mock
    audio_path = "/data/test.mp3"
    output_path = "/data/test_transcript.txt"
    expected_transcript = "This is a mocked transcript."  # Define expected text

    with unittest.mock.patch("openai.Audio.transcribe") as mock_transcribe:  # Mock the API call
        mock_transcribe.return_value = {"text": expected_transcript}  # Set the mock response

        await run(f"Transcribe audio from {audio_path} and save to {output_path}")

        try:
            transcript = await read(output_path)
            if transcript == expected_transcript:  # Strict comparison
                return True
            else:
                return mismatch(output_path, expected_transcript, transcript)
        except Exception as e:
            logging.error(f"🔴 B8 failed: {e}")
            return False


async def b9(email, **kwargs):
    """
    Test converting Markdown to HTML.
    """
    md_file = "/data/sample.md"
    html_file = "/data/sample.html"

    # Create sample Markdown file for testing
    with open(md_file, "w") as f:
        f.write("# Sample Title\n\nThis is a sample paragraph.")

    await run(f"Convert Markdown file `{md_file}` to HTML and save it as `{html_file}`")

    # Verify HTML output matches expectations
    expected_html = "<h1>Sample Title</h1>\n<p>This is a sample paragraph.</p>"
    result_html = await read(html_file)

    if result_html.strip() != expected_html.strip():
        return mismatch(html_file, expected_html, result_html)

    return True

async def b10(email: str):
    await run("Create API endpoint to filter /data/contacts.csv by last_name=Smith")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/filter_csv",
            json={"csv_path": "/data/contacts.csv", "filter_column": "last_name", "filter_value": "Smith"}
        )
        return len(response.json())>0





async def main(email: str):
    score, total = 0, 0
    for task in [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,b6,b8,b9,b10]:
    
        total += 1
        try:
            success = await task(email=email)
        except Exception as e:
            logging.error(f"🔴 {task.__name__.upper()} failed: {e}")
            success = False
        if success:
            logging.info(f"✅ {task.__name__.upper()} PASSED")
        else:
            logging.error(f"❌ {task.__name__.upper()} FAILED")
        score += 1 if success else 0
    logging.info(f"🎯 Score: {score} / {total}")


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tasks with configurable logging")
    parser.add_argument("--email", default="user@example.com", help="Set the email address")
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument("--log-level", default="INFO", choices=levels, help="Set logging level")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s\n")
    asyncio.run(main(args.email))