import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, FunctionMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if GROQ_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GROQ_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def initialize_workflow():
    web_search = TavilySearch(max_results=3) if TAVILY_API_KEY else None

    llm = ChatGroq(model="llama-3.3-70b-versatile",max_tokens=2000)

    research_agent_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a research agent. Use Tavily and any other available internet resources to fetch the latest, most relevant fitness and nutrition information for the user's needs.
- Always prioritize up-to-date, actionable data from the current internet, including:
    - Food preferences and trending diets in the user's region
    - Typical gym membership costs and home workout alternatives
    - Cost-effective, budget-friendly meal options and food prices (in INR)
    - Popular exercises and routines for the user's goal and experience
    - Local/cultural foods and substitutions
    - Any recent scientific findings or government guidelines
- For ALL food and gym costs, always provide the price in INR (₹), with links to sources (grocery sites, government, news, etc.).
- Use multiple sources if possible, and always cite links for cost data.
- Summarize findings clearly and concisely for the plan generator, with sources and links where possible.
- If you can provide a direct answer from your knowledge base or research, do so. Otherwise, indicate if further research is needed or if you are done.
'''),
        MessagesPlaceholder(variable_name="messages"),
    ])

    research_agent = create_react_agent(
        model=llm,
        tools=[web_search] if web_search else [],
        prompt=research_agent_prompt,
        name="research_agent",
    )
    
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a world-class fitness coach and nutritionist. Your task is to create a highly personalized, actionable, and safe gym training and nutrition plan.
- Use all user details and the latest research findings provided by the research agent.
- Your plan MUST be:
    - Comprehensive, well-structured, and actionable
    - Use markdown tables for all plans (especially the weekly meal plan and gym plan)
    - Include:
        # Weekly Gym Training Plan
        (Provide a markdown table: Days as rows, columns for muscle group/focus, exercises, sets, reps, rest, gym/home alternative, and estimated daily gym cost in INR. Include rest days.)
        # Weekly Meal Plan
        (Provide a markdown table: Days as rows, columns for breakfast, lunch, dinner, snacks, calories, protein, carbs, fats, and estimated daily food cost in INR. Use locally available, budget-friendly foods. All prices in INR. Add links to sources if possible.)
        # Lifestyle Tips
        (Bullet points: hydration, sleep, stress, recovery, budget tips, and any recent scientific advice. Include tips for staying on budget and using local resources.)
        # Summary
        (Markdown table: total weekly food cost (INR), total weekly gym cost (INR), weekly calorie intake, and key recommendations. List all sources/links used for cost data.)
- Do NOT include any conversational filler, introductory, or concluding remarks. Just provide the plan in structured markdown tables and bullet points.
'''),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Generate the personalized gym training and nutrition plan based on the extracted user details and research findings.")
    ])

    plan_agent = create_react_agent(
        model=llm,
        tools=[],
        prompt=plan_prompt,
        name="plan_agent",
    )

    supervisor = create_supervisor(
        model=llm,
        agents=[research_agent, plan_agent],
        prompt=(
            """You are a supervisor managing two agents: a research agent (for web search and latest info) and a plan agent (for generating the personalized plan). \
Your primary role is to ensure the user receives a perfect, well-structured, and comprehensive fitness and nutrition plan. \
Workflow:\n1. When you receive user details, first consider if the 'research_agent' needs to be invoked to gather more information related to the user's goal, diet, or health conditions before the 'plan_agent' can create a comprehensive plan.\n2. After research, ensure all relevant user data and research findings are available in the messages before handing off to the 'plan_agent'.\n3. The 'plan_agent' will then generate the final personalized plan. Your FINAL output to the user MUST be ONLY the generated plan from the 'plan_agent', without any additional conversational text or preambles from yourself. Ensure the plan is complete and well-formatted. Do not do any work yourself beyond orchestrating the agents and presenting the final plan."""
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
    return supervisor 