import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
import time
import workflow as workflow
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="FitMind AI - Gym & Diet Plan Generator",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

colored_header(
    label="FitMind AI",
    description="Your Personalized Gym & Diet Plan Generator",
    color_name="violet-70"
)


with stylable_container(
    key="main_form",
    css_styles="""
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        border-radius: 1.5rem;
        box-shadow: 0 4px 32px 0 rgba(80, 0, 120, 0.08);
        padding: 2rem 2rem 1rem 2rem;
        margin-bottom: 2rem;
    """
):
    st.markdown("## Enter Your Profile")
    with st.form("user_profile_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name", "Parthib Karak")
            age = st.number_input("Age", min_value=10, max_value=100, value=21)
            gender = st.selectbox("Gender", ["male", "female", "other"], index=0)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=64.0)
            height = st.number_input("Height (cm)", min_value=120.0, max_value=250.0, value=180.0)
            experience = st.selectbox("Experience", ["beginner", "intermediate", "advanced"], index=0)
        with col3:
            fitness_goal = st.selectbox("Fitness Goal", ["Build Muscle", "Lose Fat", "Improve Endurance", "General Fitness"], index=0)
            workout_time = st.selectbox("Workout Time", ["Morning", "Afternoon", "Evening"], index=0)
            diet = st.selectbox("Diet", ["OmniVore", "Vegetarian", "Vegan", "Keto", "Paleo"], index=0)
        allergies = st.text_input("Allergies (comma separated)", "")
        health_conditions = st.text_input("Health Conditions (comma separated)", "")
        submitted = st.form_submit_button("Generate Plan", use_container_width=True)

if submitted:
    with st.spinner("Generating your personalized plan with AI agents..."):
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
            for i, chunk in enumerate(supervisor.stream({"messages": [user_message]})):
                progress = min(100, (i+1)*10)
                progress_bar.progress(progress/100.0, text=f"AI agents are working... ({progress}%)")
                for node_name, node_content in chunk.items():
                    if node_name == "plan_agent" and "messages" in node_content:
                        for msg in reversed(node_content["messages"]):
                            if isinstance(msg, AIMessage) and msg.content and "# Weekly Gym Training Plan" in msg.content:
                                final_plan = msg.content
                                break
            progress_bar.empty()
            if final_plan:
                st.success("Here is your personalized plan!", icon="🤩")
                st.markdown(f"**User Profile:**  \nName: {name}  \nAge: {age}  \nGender: {gender}  \nWeight: {weight}kg  \nHeight: {height}cm  \nBMI: {bmi} ({bmi_status})  \nFitness Goal: {fitness_goal}  \nExperience: {experience}  \nWorkout Time: {workout_time}  \nDiet: {diet}  \nAllergies: {allergies or 'None'}  \nHealth Conditions: {health_conditions or 'None'}")
                st.markdown(final_plan)
                st.balloons()
            else:
                st.error("The AI agent did not generate a comprehensive plan. Please try again or refine your input.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.markdown(
        "<div style='text-align:center; color:#7c3aed; font-size:1.2rem; margin-top:2rem;'>"
        "Welcome to <b>FitMind AI</b>! Enter your details and get a beautiful, animated gym & diet plan instantly!"
        "</div>", unsafe_allow_html=True
    ) 