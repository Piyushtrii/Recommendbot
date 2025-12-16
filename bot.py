import streamlit as st
import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY")
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
        "skills": ["Python Basics", "Data Types", "Loops", "Functions"],
        "description": "Start programming with Python from scratch and build small projects.",
        "link": "https://www.youtube.com/watch?v=v9bOWjwdTlg"
    },
    {
        "id": "ml_201",
        "title": "Machine Learning Fundamentals",
        "topic": "machine learning",
        "level": "Intermediate",
        "skills": ["Regression", "Classification", "Model Evaluation", "Scikit-learn"],
        "description": "Learn ML concepts, algorithms, and practical applications using Python.",
        "link": "https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/"
    },
    {
        "id": "dl_301",
        "title": "Deep Learning with Python",
        "topic": "deep learning",
        "level": "Advanced",
        "skills": ["Neural Networks", "CNNs", "RNNs", "TensorFlow/PyTorch"],
        "description": "Build and train deep learning models for images and sequences.",
        "link": "https://www.udemy.com/course/deeplearning_x/"
    },
    {
        "id": "soft_101",
        "title": "Leadership and Communication Skills",
        "topic": "soft skills",
        "level": "Beginner",
        "skills": ["Teamwork", "Persuasion", "Active Listening", "Presentation"],
        "description": "Enhance workplace communication and leadership qualities.",
        "link": "https://www.udemy.com/course/soft-skills-masterclass/?couponCode=PMNVD2025"
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

Infer skill level from user's description.

Return STRICT JSON:
{
  "level": "beginner | intermediate | advanced"
}
"""

COURSE_AGENT_PROMPT = """
You are a Course Selection Agent.

Input:
- topic
- inferred_level
- course_catalog

Select best matching courses (max 3).

Return STRICT JSON:
{
  "course_ids": []
}
"""

EXPLAINER_PROMPT = """
You are SkillUpGPT, a professional learning advisor.

Explain recommended courses:
- Why the course fits the user
- Skills they will gain
- Who should take it
- Include enroll link

Use bullet points.
Friendly, agentic tone.
"""

FOLLOWUP_PROMPT = """
You are SkillUpGPT.

Answer follow-up questions about previously recommended courses.
You may compare courses, explain difficulty, outcomes, or suitability.
"""


def infer_level(skill_text):
    res = llm.invoke([
        SystemMessage(content=LEVEL_AGENT_PROMPT),
        HumanMessage(content=skill_text)
    ])
    return json.loads(res.content)["level"]

def select_courses(topic, level):
    res = llm.invoke([
        SystemMessage(content=COURSE_AGENT_PROMPT),
        HumanMessage(content=json.dumps({
            "topic": topic,
            "inferred_level": level,
            "course_catalog": DUMMY_COURSES
        }))
    ])
    ids = json.loads(res.content)["course_ids"]
    return [c for c in DUMMY_COURSES if c["id"] in ids]


def stream_llm(messages):
    placeholder = st.empty()
    full_response = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            full_response += chunk.content
            placeholder.markdown(full_response)
    return full_response


if "step" not in st.session_state:
    st.session_state.step = 0  # 0=ask topic, 1=ask skill, 99=chat

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "last_courses" not in st.session_state:
    st.session_state.last_courses = []

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("🎓 SkillUpGPT – Course Advisor")


for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

def bot(msg):
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").markdown(msg)

def user(msg):
    st.session_state.messages.append({"role": "user", "content": msg})
    st.chat_message("user").markdown(msg)


if st.session_state.step == 0:
    bot("👋 Hi! What do you want to learn today?")
    st.session_state.step = 1

user_input = st.chat_input("Type here...")

if user_input:
    user(user_input)

    # STEP 1 — Capture topic
    if st.session_state.step == 1:
        st.session_state.topic = user_input
        bot("Great 👍 Now tell me how much you already know about this topic.")
        st.session_state.step = 2

    # STEP 2 — Infer level + recommend
    elif st.session_state.step == 2:
        with st.spinner("🧠 Analyzing your skill level..."):
            level = infer_level(user_input)

        bot(f"🧠 I infer your level as **{level.capitalize()}**.")

        courses = select_courses(st.session_state.topic, level)
        st.session_state.last_courses = courses
        st.session_state.step = 99

        bot("📚 Here are the best courses for you:")

        response = stream_llm([
            SystemMessage(content=EXPLAINER_PROMPT),
            HumanMessage(content=json.dumps({
                "topic": st.session_state.topic,
                "level": level,
                "courses": courses
            }))
        ])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    #STEP — Follow-up chat
    elif st.session_state.step == 99:
        response = stream_llm([
            SystemMessage(content=FOLLOWUP_PROMPT),
            HumanMessage(content=json.dumps({
                "user_query": user_input,
                "recommended_courses": st.session_state.last_courses
            }))
        ])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
