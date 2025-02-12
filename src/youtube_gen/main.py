#!/usr/bin/env python
import sys
import warnings
import json
import time
from litellm import completion
from datetime import datetime
from youtube_gen.crew import YoutubeGen

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ✅ Fixed JSON formatting issue
raw_comments = """[
    {"video_title": "I Automated My YouTube Channel With CrewAI [Free Source Code Included]",
     "comment": "The BEST video really inspired me a lot and I can't wait to get started using CrewAI",
     "video_id": "95ab3e25-12bd-4611-b09a-4fea29846c3d",
     "comment_id": "8013f709-93af-4a06-b14a-612c988b504f"},

    {"video_title": "I Automated My YouTube Channel With CrewAI [Free Source Code Included]",
     "comment": "Very helpful video!",
     "video_id": "95ab3e25-12bd-4611-b09a-4fea29846c3d",
     "comment_id": "b12cb850-288b-40aa-82bb-0424d735fbc4"},

    {"video_title": "I Automated My YouTube Channel With CrewAI [Free Source Code Included]",
     "comment": "Awesome video Bran. Is it possible to research popular Reddit posts on a particular topic based on user engagement?",
     "video_id": "95ab3e25-12bd-4611-b09a-4fea29846c3d",
     "comment_id": "a4789124-2ebb-4180-887c-6d0934d69324"}
]"""  # ✅ Shortened for clarity, make sure to use the full version.

def chunk_comments(comments, max_tokens=3000):
    """
    Split comments into batches, ensuring they stay within the Groq token limit.
    """
    chunk = []
    total_tokens = 0
    for comment in comments:
        comment_tokens = len(comment["comment"].split())  # Approximate token count
        if total_tokens + comment_tokens > max_tokens:
            yield chunk
            chunk = []
            total_tokens = 0
        chunk.append(comment)
        total_tokens += comment_tokens
    if chunk:
        yield chunk

def run():
    """
    Run the CrewAI pipeline with chunked input to avoid token limit issues.
    """
    try:
        comments_list = json.loads(raw_comments)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    for idx, comment_batch in enumerate(chunk_comments(comments_list, max_tokens=3000)):
        print(f"Processing batch {idx + 1}...")

        # Prepare input for CrewAI
        inputs = {"comments": json.dumps(comment_batch)}

        try:
            YoutubeGen().crew().kickoff(inputs=inputs)
        except Exception as e:
            print(f"Error in batch {idx + 1}: {e}")

        time.sleep(15)  # Adjust delay as needed

def train():
    """
    Train the crew for a given number of iterations.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py train <n_iterations> <filename>")
        return
    
    try:
        n_iterations = int(sys.argv[1])
        filename = sys.argv[2]
    except ValueError:
        print("Error: n_iterations must be an integer.")
        return

    inputs = {"topic": "AI LLMs"}
    try:
        YoutubeGen().crew().train(n_iterations=n_iterations, filename=filename, inputs=inputs)
    except Exception as e:
        print(f"Error during training: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py replay <task_id>")
        return

    try:
        YoutubeGen().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        print(f"Error during replay: {e}")

def test():
    """
    Test the crew execution and return the results.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py test <n_iterations> <openai_model_name>")
        return

    try:
        n_iterations = int(sys.argv[1])
        model_name = sys.argv[2]
    except ValueError:
        print("Error: n_iterations must be an integer.")
        return

    inputs = {"topic": "AI LLMs"}
    try:
        YoutubeGen().crew().test(n_iterations=n_iterations, openai_model_name=model_name, inputs=inputs)
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <run|train|replay|test>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
