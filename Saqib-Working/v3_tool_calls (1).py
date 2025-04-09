import asyncio
import json
from typing import List
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import os
import google.generativeai as genai
from agents import ModelSettings

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel(os.getenv('MODEL', 'models/gemini-1.5-pro-latest'))
model = ModelSettings(model_name=gemini_model.model_name)

# --- Models for structured outputs ---

class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    budget: float
    activities: List[str] = Field(description="List of recommended activities")
    notes: str = Field(description="Additional notes or recommendations")

# --- Tools ---

@function_tool
def get_weather_forecast(city: str, date: str) -> str:
    """Get the weather forecast for a city on a specific date."""
    # This tool is defined but not currently integrated into the simplified agents.py run method
    weather_data = {
        "New York": {"sunny": 0.3, "rainy": 0.4, "cloudy": 0.3},
        "Los Angeles": {"sunny": 0.8, "rainy": 0.1, "cloudy": 0.1},
        "Chicago": {"sunny": 0.4, "rainy": 0.3, "cloudy": 0.3},
        "Murree": {"sunny": 0.2, "rainy": 0.6, "cloudy": 0.2},
        "London": {"sunny": 0.2, "rainy": 0.5, "cloudy": 0.3},
        "Paris": {"sunny": 0.4, "rainy": 0.3, "cloudy": 0.3},
        "Tokyo": {"sunny": 0.5, "rainy": 0.3, "cloudy": 0.2},
    }

    if city in weather_data:
        conditions = weather_data[city]
        highest_prob = max(conditions, key=conditions.get)
        temp_range = {
            "New York": "15-25°C",
            "Los Angeles": "20-30°C",
            "Chicago": "10-20°C",
            "Murree": "0-5°C",
            "London": "10-18°C",
            "Paris": "12-22°C",
            "Tokyo": "15-25°C",
        }
        return f"The weather in {city} on {date} is forecasted to be {highest_prob} with temperatures around {temp_range.get(city, '15-25°C')}."
    else:
        return f"Weather forecast for {city} is not available."

# --- Main Travel Agent ---

travel_agent = Agent(
    name="Travel Planner",
    instructions="""
    You are a comprehensive travel planning assistant that helps users plan their perfect trip.
    You MUST use the get_weather_forecast tool to get weather information for destinations.
    Always be helpful, informative, and enthusiastic about travel. Provide specific recommendations
    based on the user's interests and preferences. Consider weather, attractions, budget, and duration.
    
    IMPORTANT INSTRUCTIONS:
    1. For any destination mentioned, you MUST call the get_weather_forecast tool to get weather information
    2. Include the weather information in the notes section of your response
    3. You MUST respond with a valid JSON object that includes:
       - destination: The name of the destination
       - duration_days: The number of days as an integer
       - budget: The budget as a number (float)
       - activities: An array of strings describing recommended activities
       - notes: A string containing additional recommendations and weather information
    
    Do not include any text outside the JSON object. The response must be valid JSON that can be parsed.
    """,
    model=model,
    tools=[get_weather_forecast],
    output_type=TravelPlan
)

# --- Main Function ---

async def main():
    queries = [
        "I'm planning a trip to Murree Hills for 3 days with a budget of 20000 PKR. What should I do there and what is the weather going to look like?"
    ]

    for query in queries:
        print("\n" + "="*150)
        print(f"QUERY: {query}")

        # Get the result directly from the agent
        print("Executing Runner.run in v3_tool_calls.py...")  # Verification print
        result = await Runner.run(travel_agent, query)

        print("\nFINAL RESPONSE:")
        print(f"Result type: {type(result)}")  # Debug print
        
        # Handle both direct TravelPlan and RunResult cases
        if isinstance(result, TravelPlan):
            travel_plan = result
        elif hasattr(result, 'final_output') and isinstance(result.final_output, TravelPlan):
            travel_plan = result.final_output
            print("\n=== FORMATTED TRAVEL PLAN ===")
            print(f"\n🌍 TRAVEL PLAN FOR {travel_plan.destination.upper()} 🌍")
            print(f"Duration: {travel_plan.duration_days} days")
            print(f"Budget: PKR {travel_plan.budget:,.2f}")

            print("\n🎯 RECOMMENDED ACTIVITIES:")
            if travel_plan.activities:
                for i, activity in enumerate(travel_plan.activities, 1):
                    print(f"  {i}. {activity}")
            else:
                print("  No activities recommended.")

            print(f"\n📝 NOTES:")
            print(f"  {travel_plan.notes}")
            print("\n=== END FORMATTED PLAN ===")
        else:
            print("Error: Received unexpected result format from agent.")
            print("Raw response:", result)  # Print the raw result for debugging

    # Add a longer delay to allow background tasks (like gRPC cleanup) to complete more reliably
    await asyncio.sleep(1.0)

if __name__ == "__main__":
    asyncio.run(main())
