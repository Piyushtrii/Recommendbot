import streamlit as st
import json
import re  
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY - Check Streamlit Secrets")
    st.stop()

st.set_page_config(page_title="SkillUpGPT - Course Advisor", layout="wide")


@st.cache_resource
def init_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=0.3,
        streaming=True
    )

llm = init_llm()


DUMMY_COURSES = [
    {
        "id": "py_101",
        "title": "Python Programming for Beginners",
        "topic": "python",
        "level": "Beginner",
        "skills": ["Python Basics", "Loops", "Functions"],
        "description": "Start programming with Python from scratch.",
        "link": "https://www.youtube.com/watch?v=v9bOWjwdTlg"
    },
    {
        "id": "ml_201",
        "title": "Machine Learning Fundamentals",
        "topic": "machine learning",
        "level": "Intermediate",
        "skills": ["Regression", "Classification", "Scikit-learn"],
        "description": "Learn ML algorithms with Python.",
        "link": "https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/"
    },
    {
        "id": "dl_301",
        "title": "Deep Learning with Python",
        "topic": "deep learning",
        "level": "Advanced",
        "skills": ["CNNs", "RNNs", "PyTorch"],
        "description": "Build deep learning models.",
        "link": "https://www.udemy.com/course/deeplearning_x/"
    },
    {
        "id": "soft_101",
        "title": "Leadership and Communication Skills",
        "topic": "soft skills",
        "level": "Beginner",
        "skills": ["Communication", "Teamwork"],
        "description": "Improve workplace communication.",
        "link": "https://www.udemy.com/course/soft-skills-masterclass/"
    },
     {
        "id": "soft_201",
        "title": "Emotional Intelligence at Work",
        "topic": "soft skills",
        "level": "Intermediate",
        "skills": ["Self-awareness", "Empathy", "Conflict Resolution"],
        "description": "Learn to manage emotions and improve interpersonal skills at work.",
        "link": "https://www.udemy.com/course/soft-skills-the-11-essential-career-soft-skills/"
    },
]


LEVEL_AGENT_PROMPT = """
You are a Skill Level Inference Agent.

Infer skill level from user description.

Return STRICT JSON:
{
  "level": "beginner | intermediate | advanced"
}
"""

PLANNER_PROMPT = """
You are a Planner Agent.

Decide the next action based on current state.

Actions:
- ask_topic
- ask_skill
- recommend_courses
- follow_up

Return STRICT JSON:
{
  "action": ""
}
"""

EXPLAINER_PROMPT = """
You are SkillUpGPT, a professional learning advisor.

Explain recommended courses:
- Why it fits
- Skills gained
- Who should take it
- Include course link

Use bullet points.
"""

FOLLOWUP_PROMPT = """
You are SkillUpGPT.

Answer follow-up questions using previous recommendations.
"""


def get_llm_memory(last_n=6):
    memory = []
    for m in st.session_state.messages[-last_n:]:
        if m["role"] == "user":
            memory.append(HumanMessage(content=m["content"]))
        else:
            memory.append(SystemMessage(content=m["content"]))
    return memory


def course_search_tool(topic, level):
    return [
        c for c in DUMMY_COURSES
        if topic.lower() in c["topic"].lower()
        and c["level"].lower() == level.lower()
    ]


def level_agent(skill_text):
    res = llm.invoke([
        SystemMessage(content=LEVEL_AGENT_PROMPT),
        HumanMessage(content=skill_text)
    ])
    #parsing with fallback
    try:
        return json.loads(res.content)["level"].lower()
    except:
        level_match = re.search(r'(beginner|intermediate|advanced)', res.content, re.IGNORECASE)
        return level_match.group(1).lower() if level_match else "intermediate"

def planner_agent():
    res = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=json.dumps({
            "topic": st.session_state.topic,
            "last_courses": st.session_state.last_courses
        }))
    ])
    try:
        return json.loads(res.content)["action"]
    except:
        return "follow_up"  #default setback


def tool_agent(topic, level):
    courses = [c for c in DUMMY_COURSES
               if topic.lower() in c["topic"].lower()
               and c["level"].lower() == level.lower()]
    return courses if courses else DUMMY_COURSES[:1] 

def explainer_agent(topic, level, courses):
    return stream_llm([
        SystemMessage(content=EXPLAINER_PROMPT),
        HumanMessage(content=json.dumps({
            "topic": topic,
            "level": level,
            "courses": courses
        }))
    ])


def stream_llm(messages):
    placeholder = st.empty()
    full = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            full += chunk.content
            placeholder.markdown(full)
    return full


if "messages" not in st.session_state:
    st.session_state.messages = []

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "last_courses" not in st.session_state:
    st.session_state.last_courses = []

if "awaiting_skill" not in st.session_state:
    st.session_state.awaiting_skill = False

st.title("🎓 SkillUpGPT – Agentic Course Advisor")

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

def bot(msg):
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").markdown(msg)

def user(msg):
    st.session_state.messages.append({"role": "user", "content": msg})
    st.chat_message("user").markdown(msg)

if len(st.session_state.messages) == 0:
    bot("👋 Hi! What do you want to learn today?")

user_input = st.chat_input("Type here...")

if user_input:
    user(user_input)

    #Topic capture
    if not st.session_state.topic:
        st.session_state.topic = user_input
        st.session_state.awaiting_skill = True
        bot("Great 👍 Tell me how much you already know about this topic.")

    #Skill capture + recommendation
    elif st.session_state.awaiting_skill:
        level = level_agent(user_input)
        bot(f"🧠 I infer your level as **{level.capitalize()}**.")

        courses = tool_agent(st.session_state.topic, level)
        st.session_state.last_courses = courses
        st.session_state.awaiting_skill = False

        bot("📚 Here are the best courses for you:")
        response = explainer_agent(
            st.session_state.topic,
            level,
            courses
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    #Follow-up chat
    else:
        response = stream_llm(
            get_llm_memory() + [
                SystemMessage(content=FOLLOWUP_PROMPT),
                HumanMessage(content=user_input)
            ]
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": response

        })
