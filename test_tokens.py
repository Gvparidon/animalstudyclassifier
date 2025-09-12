import json
import tiktoken

# Path to your JSON file
jsonl_file_path = "azure/batches/batch_20250912-1217_1.jsonl"

# Choose the encoding for the model you are using
# For example: "gpt-3.5-turbo" or "gpt-4"
model_name = "gpt-4.1"
# Initialize tiktoken encoding for the model
encoding = tiktoken.encoding_for_model(model_name)

user_contents = []

with open(jsonl_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        messages = data.get("body", {}).get("messages", [])
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if content is None:
                    content = ""  # Make sure it's a string
                elif not isinstance(content, str):
                    content = str(content)
                user_contents.append(content)
            if message.get("role") == "system":
                content = message.get("content")
                if content is None:
                    content = ""  # Make sure it's a string
                elif not isinstance(content, str):
                    content = str(content)
                user_contents.append(content)
            

# Count tokens for each user content
total_tokens = 0
amount = 0
for content in user_contents:
    tokens = len(encoding.encode(content))
    if tokens == 0:
        amount += 1
    total_tokens += tokens

print(f"\nTotal tokens across all user messages: {total_tokens}")
print(f"Amount of empty messages: {amount}")