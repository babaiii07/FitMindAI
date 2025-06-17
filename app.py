import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, FunctionMessage
from langchain_core.tools import tool
from workflow import initialize_workflow

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set API keys for libraries
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize the LangGraph supervisor workflow
supervisor = initialize_workflow()

# --- Data Model for API ---
class PlanRequest(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    height: float
    goal: str
    experience: str
    workout_time: str
    diet_preference: str
    allergies: str = "None"
    health_conditions: str = "None"

# --- BMI Calculation ---
def calculate_bmi(weight_kg, height_cm):
    try:
        weight_kg = float(weight_kg)
        height_cm = float(height_cm)
    except ValueError:
        return None, "Invalid number format for weight or height."
    if height_cm <= 0:
        return None, "Height must be greater than 0."
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 24.9:
        category = "Normal weight"
    elif bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obesity"
    return round(bmi, 2), category

# --- AGENTS SETUP ---
# 1. Research Agent (Tavily)
web_search = TavilySearch(max_results=3) if TAVILY_API_KEY else None

# 2. Plan Generator Agent (Groq LLM)
llm = ChatGroq(model_name="gemma2-9b-it", max_tokens=5000)

# Research Agent Definition
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

# Plan Generator Agent (Prompt now includes instructions for parsing messages)
plan_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        """You are a world-class fitness coach and nutritionist. Your task is to create a highly personalized, actionable, and safe gym training and nutrition plan. 
        You will receive the user's details and potentially research findings in the conversation history within the 'messages' variable. 
        Carefully extract all necessary information from the `HumanMessage` that starts with "User details:" and any `AIMessage` or `FunctionMessage` from the `research_agent` or tools. 
        Ensure all required fields for the plan are identified from these messages. 
        
        Your output MUST be a comprehensive, well-structured fitness and nutrition plan, using clear markdown headings and bullet points. DO NOT include any conversational filler, introductory, or concluding remarks. Just provide the plan. The plan should include:
        
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
    MessagesPlaceholder(variable_name="messages"), # This will contain the conversation history
    ("human", "Generate the personalized gym training and nutrition plan based on the extracted user details and research findings.")
])

plan_agent = create_react_agent(
    model=llm,
    tools=[], # No tools for this agent
    prompt=plan_prompt, # This is the updated prompt
    name="plan_agent",
)

# Supervisor Agent
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

# --- API Endpoints (Flask) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    try:
        data = request.get_json()
        req = PlanRequest(**data)
        bmi, bmi_category = calculate_bmi(req.weight, req.height)
        if bmi is None:
            return jsonify({"error": bmi_category}), 400

        # Prepare initial message for the supervisor
        user_message_content = (
            f"User details:\n"
            f"Name: {req.name}\nAge: {req.age}\nGender: {req.gender}\nWeight: {req.weight} kg\nHeight: {req.height} cm\n"
            f"BMI: {bmi} ({bmi_category})\nGoal: {req.goal}\nExperience: {req.experience}\nWorkout Time: {req.workout_time}\n"
            f"Diet Preference: {req.diet_preference}\nAllergies: {req.allergies}\nHealth Conditions: {req.health_conditions}"
        )
        user_message = HumanMessage(content=user_message_content)

        # Run the LangGraph workflow
        final_plan = ""
        for chunk in supervisor.stream({"messages": [user_message]}):
            # Iterate through messages in each chunk to find the plan_agent's output
            for node_name, node_content in chunk.items():
                if node_name == "plan_agent" and "messages" in node_content:
                    for msg in reversed(node_content["messages"]):
                        if isinstance(msg, AIMessage) and msg.content and "## Weekly Gym Training Plan" in msg.content: # Look for the main plan message
                            final_plan = msg.content
                            break # Found the plan, exit inner loop
                if final_plan: # If plan found, exit outer loop as well
                    break

        if not final_plan:
            return jsonify({"error": "The AI agent did not generate a comprehensive plan. This could be due to an issue in the agent workflow or insufficient information. Please try again or refine your input."}), 500

        return jsonify({"plan": final_plan})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


