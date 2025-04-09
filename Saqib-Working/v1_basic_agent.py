from agents import Agent, Runner
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

agent = Agent(
    name="Tour Guide", 
    instructions="""
        you are an expert tour guide.
        you are given the likes and dislikes of the user.
        you are also given the budget of the user.
        you are also given the country where the user wants to travel.
        you need to suggest a destination to the user based on their likes and dislikes.
        you need to provide a detailed itinerary for the destination.
        you need to provide the best time to visit the destination.
        you need to provide the best way to travel to the destination.
        you need to provide the best places to visit in the destination.
        you need to provide the best restaurants to visit in the destination.
        you need to provide the best hotels to visit in the destination.
        you need to provide the best activities to do in the destination.        
    """,
    model="gpt-4o-mini"
)

def main():
    user_input = """
        Plan a 2 day trip for the user in Pakistan.
        The user likes deserts, rushy areas and historical places.
        The user's budget is 5000 Pakistani Rupees per day.        
        """
    result = Runner.run_sync(agent, user_input)
    print(result.final_output)

if __name__ == "__main__":
    main()