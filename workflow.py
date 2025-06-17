import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, FunctionMessage
from langchain_core.tools import tool

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def initialize_workflow():
    web_search = TavilySearch(max_results=3) if TAVILY_API_KEY else None

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", max_tokens=5000)

    research_agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research agent. Use Tavily to fetch the latest, most relevant fitness and nutrition information for the user's needs. Summarize findings clearly and concisely for the plan generator. If you can provide a direct answer from your knowledge base or research, do so. Otherwise, indicate if further research is needed or if you are done."""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    research_agent = create_react_agent(
        model=llm,
        tools=[web_search] if web_search else [],
        prompt=research_agent_prompt,
        name="research_agent",
    )
    
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", 
            """You are a world-class fitness coach and nutritionist. Your task is to create a highly personalized, actionable, and safe gym training and nutrition plan. 
            You will receive the user's details and potentially research findings in the conversation history within the 'messages' variable. 
            Carefully extract all necessary information from the `HumanMessage` that starts with "User details:" and any `AIMessage` or `FunctionMessage` from the `research_agent` or tools. 
            Ensure all required fields for the plan are identified from these messages. 
            
            Your output MUST be a comprehensive, well-structured fitness and nutrition plan, using clear markdown headings and bullet points. DO NOT include any conversational filler, introductory, or concluding remarks. Just provide the plan.
            The plan should include:
            
            # Weekly Gym Training Plan
            (Detailed exercises with sets, reps, rest for each day of the week, including rest days)
            
            # Daily Meal Plan
            (Detailed meal plan for each day, covering breakfast, lunch, dinner, and snacks, with nutrition breakdown and food suggestions)
            
            # Lifestyle Tips
            (Practical tips on hydration, sleep, stress management, etc.)
            
            # Summary
            (Key highlights and main recommendations of the plan)
            
            If you used any recent research or web data (from the research agent), ensure it's cited within the plan. If any information seems missing or unclear from the input messages, state it clearly in a brief note at the beginning of your plan, but proceed with the plan using the best available information."""
        ),
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
            """You are a supervisor managing two agents: a research agent (for web search and latest info) and a plan agent (for generating the personalized plan). 
            Your primary role is to ensure the user receives a perfect, well-structured, and comprehensive fitness and nutrition plan. 
            
            Workflow:
            1. When you receive user details, first consider if the 'research_agent' needs to be invoked to gather more information related to the user's goal, diet, or health conditions before the 'plan_agent' can create a comprehensive plan.
            2. After research, ensure all relevant user data and research findings are available in the messages before handing off to the 'plan_agent'.
            3. The 'plan_agent' will then generate the final personalized plan. Your FINAL output to the user MUST be ONLY the generated plan from the 'plan_agent', without any additional conversational text or preambles from yourself. Ensure the plan is complete and well-formatted. Do not do any work yourself beyond orchestrating the agents and presenting the final plan."""
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
    return supervisor 