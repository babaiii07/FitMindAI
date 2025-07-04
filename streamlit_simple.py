import streamlit as st
import workflow
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="FitMind AI - Modern Gym & Diet Planner",
    page_icon="🏋️‍♂️",
    layout="centered"
)

st.markdown("""
# 🏋️‍♂️ FitMind AI
#### Your Modern Gym & Diet Plan Generator
""")

# Use session state to clear fields after submit
if 'clear_form' not in st.session_state:
    st.session_state.clear_form = False

def reset_form():
    st.session_state.clear_form = True

with st.form("user_profile_form", clear_on_submit=True):
    st.subheader("Personal Details")
    name = st.text_input("Name", "" if st.session_state.clear_form else "")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=21 if not st.session_state.clear_form else 21)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
        experience = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], index=0)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=64.0 if not st.session_state.clear_form else 64.0)
        height = st.number_input("Height (cm)", min_value=120.0, max_value=250.0, value=180.0 if not st.session_state.clear_form else 180.0)
        workout_time = st.selectbox("Workout Time", ["Morning", "Afternoon", "Evening"], index=0)
    st.subheader("Goals & Preferences")
    fitness_goal = st.selectbox("Fitness Goal", ["Build Muscle", "Lose Fat", "Improve Endurance", "General Fitness"], index=0)
    diet = st.selectbox("Diet", ["Omnivore", "Vegetarian", "Vegan", "Keto", "Paleo"], index=0)
    allergies = st.text_input("Allergies (comma separated)", "")
    health_conditions = st.text_input("Health Conditions (comma separated)", "")
    submitted = st.form_submit_button("✨ Generate My Plan!", on_click=reset_form)

if submitted:
    # Validate height and weight to avoid ZeroDivisionError
    if height <= 0 or weight <= 0:
        st.error("Please enter valid, non-zero values for height and weight.")
    else:
        with st.spinner("Generating your personalized plan..."):
            bmi = round(weight / ((height / 100) ** 2), 2)
            bmi_status = (
                "Underweight" if bmi < 18.5 else
                "Normal weight" if bmi < 25 else
                "Overweight" if bmi < 30 else
                "Obese"
            )
            user_message_content = (
                f"User Profile: Name: {name}, Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm, BMI: {bmi} ({bmi_status}).\n"
                f"Fitness Goal: {fitness_goal}. Experience: {experience}. Workout Time: {workout_time}.\n"
                f"Diet: {diet}. Allergies: {allergies or 'None'}. Health Conditions: {health_conditions or 'None'}.\n"
                f"Please generate a comprehensive weekly gym training and diet plan based on this profile."
            )
            user_message = HumanMessage(content=user_message_content)
            supervisor = workflow.initialize_workflow()
            final_plan = ""
            try:
                progress_bar = st.progress(0, text="AI agents are working...")
                all_chunks = []
                for i, chunk in enumerate(supervisor.stream({"messages": [user_message]})):
                    progress = min(100, (i+1)*10)
                    progress_bar.progress(progress/100.0, text=f"AI agents are working... ({progress}%)")
                    all_chunks.append(chunk)
                progress_bar.empty()
                for chunk in all_chunks:
                    for node_name, node_content in chunk.items():
                        if node_name == "plan_agent" and "messages" in node_content:
                            for msg in reversed(node_content["messages"]):
                                if isinstance(msg, AIMessage) and msg.content and (
                                    "Weekly Gym" in msg.content or "Meal Plan" in msg.content or "Plan" in msg.content
                                ):
                                    final_plan = msg.content
                                    break
                if final_plan:
                    st.success("Here is your personalized plan!", icon="🤩")
                    with st.expander("Show User Profile"):
                        st.markdown(f"**Name:** {name}\n\n**Age:** {age}\n\n**Gender:** {gender}\n\n**Weight:** {weight}kg\n\n**Height:** {height}cm\n\n**BMI:** {bmi} ({bmi_status})\n\n**Fitness Goal:** {fitness_goal}\n\n**Experience:** {experience}\n\n**Workout Time:** {workout_time}\n\n**Diet:** {diet}\n\n**Allergies:** {allergies or 'None'}\n\n**Health Conditions:** {health_conditions or 'None'}")
                    st.markdown(final_plan)
                else:
                    st.error("The AI agent did not generate a comprehensive plan. Please try again or refine your input.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    pass