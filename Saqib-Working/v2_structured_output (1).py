import asyncio
import os
from agents import Agent, Runner, ModelSettings # Import ModelSettings
from typing import List
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model_name = os.getenv('MODEL', 'gemini-1.5-flash') # Default to a known model if not set
travel_model_settings = ModelSettings(
    model_name=model_name,
    temperature=0.7, # Default or adjust as needed
    max_tokens=1024, # Default or adjust as needed    
)
# --- Models for structured outputs ---
class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    budget: float
    activities: List[str] = Field(description="List of recommended activities")
    notes: str = Field(description="Additional notes or recommendations")
# --- Main Travel Agent ---
travel_agent = Agent(
    name="Travel Planner",
    instructions="""
    You are a comprehensive travel planning assistant that helps users plan their perfect trip.
    You can create personalized travel itineraries based on the user's interests and preferences.
    Always be helpful, informative, and enthusiastic about travel. Provide specific recommendations based on the user's interests and preferences.
    When creating travel plans, consider:
    - Local attractions and activities
    - Budget constraints
    - Travel duration
    """,
    model=travel_model_settings, # Pass the ModelSettings instance
    output_type=TravelPlan
)
# --- Main Function ---
async def main():
    # Example queries to test the system
    queries = [        
        "I want to visit Naran Kaghan (Northern Areas of Pakistan) for a week with a budget of 50000 PKR. What activities do you recommend?",
        "I'm planning a trip to Lahore (Punjab, Pakistan) for 1 day with a budget of 5000 PKR. What should I do there?",
        "I'm planning a trip to Faisalabad (Punjab, Pakistan) for 1 day with a budget of 100 PKR. What should I do there?"
    ]
    for query in queries:
        print("\n" + "="*150)
        print(f"QUERY: {query}")        
        result = await Runner.run(travel_agent, query)        
        print("\nFINAL RESPONSE:")
        travel_plan = result.final_output
        
        # Format the output in a nicer way
        print(f"\n🌍 TRAVEL PLAN FOR {travel_plan.destination.upper()} 🌍")
        print(f"Duration: {travel_plan.duration_days} days")
        print(f"Budget: {travel_plan.budget} PKR")        
        print("\n🎯 RECOMMENDED ACTIVITIES:")
        for i, activity in enumerate(travel_plan.activities, 1):
            print(f"  {i}. {activity}")        
        print(f"\n📝 NOTES: {travel_plan.notes}")
if __name__ == "__main__":
    asyncio.run(main())