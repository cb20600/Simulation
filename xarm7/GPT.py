import os
from openai import OpenAI
from dotenv import load_dotenv

# # 初始化 API
# load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # 接收用户指令
# user_input = input("Enter your robot instruction: ")

def parser(user_input):
    # LLM 理解指令并输出任务步骤（自然语言描述）
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot assistant that understands natural language instructions.\n"
                "Your job is to interpret the user's task command and output a **step-by-step subtask plan**.\n"
                "Each step should be described clearly in natural language, including:\n"
                "- what to do\n"
                "- which objects are involved\n"
                "- what the goal is\n"
                "- any spatial relations or constraints\n\n"
                "Output should be plain text only, no JSON or code formatting.\n"
                "Example:\n"
                "1. Move the apple close to the banana.\n"
                "2. Rotate the apple to align with the banana.\n"
                "3. Place the potato between the banana and the orange."
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    # 打印LLM输出的任务计划
    print("\n🧠 Subtask Plan:")
    print(response.choices[0].message.content)

    return response.choices[0].message.content

def trajectory_plan(subtask_txt_path="llm_subtasks.txt", grasp_json_path="grasp_infos.json"):
    """
    根据 LLM 子任务描述和感知到的物体抓取信息，调用 LLM 生成一系列轨迹点动作计划。

    参数:
        subtask_txt_path (str): 包含 LLM 输出子任务描述的 .txt 文件路径
        grasp_json_path (str): 包含感知信息（中心点、角度、宽度）的 .json 文件路径

    返回:
        trajectory_plan (str): LLM 输出的自然语言轨迹点动作序列
    """

    # 读取子任务描述
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    # 读取感知到的物体抓取信息
    with open(grasp_json_path, "r") as f:
        grasp_data = json.load(f)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot trajectory planner. Based on the user's task breakdown and the sensed object information, "
                "generate a detailed motion plan for the robot arm to follow.\n"
                "Each action should include:\n"
                "- A step index\n"
                "- Target 3D position (x, y, z)\n"
                "- Target orientation (quaternion: x, y, z, w)\n"
                "- Gripper action: open or close (with value)\n"
                "- Description of what this step does\n\n"
                "Output in clear numbered steps in plain text.\n"
                "Avoid JSON or markdown formatting.\n"
                "The grasp_data contains: object index, center (u,v), grasp angle (degrees), width (meters). "
                "You may assume a mapping from (u,v) image center to (x,y,z) world coordinates is already done."
            )
        },
        {
            "role": "user",
            "content": f"Task description:\n{subtask_text.strip()}\n\nSensed grasp data:\n{json.dumps(grasp_data, indent=2)}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    result = response.choices[0].message.content
    print("🧠 LLM Trajectory Plan:\n")
    print(result)

    return result
