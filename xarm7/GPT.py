import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("✅ OpenAI key loaded:", os.getenv("OPENAI_API_KEY") is not None)

def chat_with_gpt(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )
    return response.choices[0].message

print("💬 Start task planning (type 'q' to quit)")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["q", "quit", "exit"]:
        print("👋 Goodbye!")
        break

    # Stage 1: extract object-action-target
    stage1_messages = [{
        "role": "system",
        "content": (
            "You are an expert at interpreting robot control instructions.\n"
            "The user will give you a natural language command, and your job is to extract a structured list of triplets.\n"
            "Each item should contain:\n"
            "- object: the object to manipulate\n"
            "- action: move / rotate / align / place_near\n"
            "- target: bin name, reference object, or direction info\n"
            "Output a JSON array of triplets."
        )
    }, {
        "role": "user",
        "content": user_input
    }]

    stage1_result = chat_with_gpt(stage1_messages)

    try:
        triplets = json.loads(stage1_result.content)
        print("\n🔎 Parsed triplets:")
        print(json.dumps(triplets, indent=2, ensure_ascii=False))
    except Exception as e:
        print("⚠️ Failed to parse Stage 1 output:", e)
        print("Raw output:", stage1_result.content)
        continue

    # Stage 2: expand into subtask plan
    stage2_messages = [{
        "role": "system",
        "content": (
            "You are a robot task planner.\n"
            "Given a list of structured object-action-target triplets, generate a list of subtasks.\n"
            "Each subtask should include:\n"
            "- task_id: index starting from 1\n"
            "- description: natural language description\n"
            "- action: same as input (move, align, etc.)\n"
            "- object_name, target_bin, ref_object, direction, etc.\n"
            "Output a JSON array of subtasks ready for planning."
        )
    }, {
        "role": "user",
        "content": json.dumps(triplets, ensure_ascii=False)
    }]

    stage2_result = chat_with_gpt(stage2_messages)

    try:
        subtasks = json.loads(stage2_result.content)
        print("\n✅ Final Subtask Plan:")
        print(json.dumps(subtasks, indent=2, ensure_ascii=False))

        # Optionally save
        save = input("💾 Save as subtask_plan.json? (y/n): ")
        if save.lower() == "y":
            with open("subtask_plan.json", "w") as f:
                json.dump(subtasks, f, indent=2, ensure_ascii=False)
            print("✅ Saved as subtask_plan.json")
    except Exception as e:
        print("⚠️ Failed to parse Stage 2 output:", e)
        print("Raw output:", stage2_result.content)
