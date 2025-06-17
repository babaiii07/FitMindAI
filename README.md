# FitMind AI: Multi-Agentic Fitness & Nutrition Planner

This project is a next-generation fitness and nutrition planner powered by a multi-agent workflow using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), [Groq LLM](https://groq.com/), and [Tavily](https://www.tavily.com/) for real-time research.

## Features
- **Supervisor Multi-Agent Workflow**: Orchestrates research and plan generation agents for robust, up-to-date plans.
- **Research Agent**: Fetches latest fitness/nutrition info using Tavily web search.
- **Plan Generator Agent**: Uses Groq LLM to create highly personalized, actionable plans.
- **Flask Backend**: A lightweight Python web framework for seamless frontend integration.
- **Frontend Compatibility**: Works with the existing HTML/JS frontend (no changes needed to the contract).

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables**:
   - Create a `.env` file in the project root with:
     ```
     GROQ_API_KEY=your_groq_api_key
     TAVILY_API_KEY=your_tavily_api_key
     ```
   - (Tavily is optional; fallback to DuckDuckGo can be added if needed)

## Running the Application

1.  **Backend (Flask)**:
    ```bash
    python app.py
    ```
    The backend will run on `http://127.0.0.1:5000`.

2.  **Frontend (HTML/JS)**:
    Open `templates/index.html` in your web browser.

### Running with Docker

1.  **Build the Docker image**:
    ```bash
    docker build -t fitmind-ai .
    ```

2.  **Run the Docker container**:
    ```bash
    docker run -p 5000:5000 fitmind-ai
    ```
    This will make the Flask backend accessible at `http://localhost:5000`.

## API Usage

**Endpoint**: `/generate_plan` (POST)
**Content-Type**: `application/json`

**Request Body Example**:

```json
{
    "name": "John Doe",
    "age": 30,
    "gender": "Male",
    "weight": 75.0,
    "height": 175.0,
    "goal": "Build muscle and lose fat",
    "experience": "Intermediate",
    "workout_time": "Evening",
    "diet_preference": "Vegetarian",
    "allergies": "None",
    "health_conditions": "None"
}
```

**Response Example (Success)**:

```json
{
    "plan": "# Weekly Gym Training Plan\n...\n# Daily Meal Plan\n...\n# Lifestyle Tips\n...\n# Summary\n..."
}
```

**Response Example (Error)**:

```json
{
    "error": "Error message details"
}
```

## Credits

This application is powered by:

*   **Flask**: Web framework for the backend.
*   **LangGraph**: For building robust, multi-agent LLM applications.
*   **LangChain**: For integrating with various LLMs and tools.
*   **Groq**: High-performance LLM for generating plans (using `gemma2-9b-it`).
*   **Tavily**: For intelligent web search capabilities.

---
*For questions or contributions, please open an issue or pull request.* 