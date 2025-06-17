import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from workflow import initialize_workflow

app = Flask(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

supervisor = initialize_workflow()

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


