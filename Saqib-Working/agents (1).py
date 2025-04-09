from typing import List, Optional, Any, Type, Callable, Generic, TypeVar
from pydantic import BaseModel
import google.generativeai as genai
import json
import asyncio
from functools import wraps
from dataclasses import dataclass

TContext = TypeVar('TContext')

@dataclass
class RunContextWrapper(Generic[TContext]):
    context: TContext

@dataclass
class ModelSettings:
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

@dataclass
class GuardrailFunctionOutput:
    output_info: Any
    tripwire_triggered: bool

class InputGuardrailTripwireTriggered(Exception):
    pass

class InputGuardrail(Generic[TContext]):
    def __init__(self, guardrail_function: Callable[[TContext, Any, Any], GuardrailFunctionOutput]):
        self.guardrail_function = guardrail_function

@dataclass
class RunResult:
    final_output: Any
    raw_responses: List[str] = None
    new_items: List[Any] = None
    input_guardrail_results: List[Any] = None
    output_guardrail_results: List[Any] = None

def function_tool(func: Callable):
    """Decorator to mark a function as a tool for the agent."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.is_tool = True
    return wrapper

class Agent(Generic[TContext]):
    def __init__(
        self,
        name: str,
        instructions: str,
        model: ModelSettings,
        tools: List[Callable] = None,
        handoffs: List['Agent'] = None,
        output_type: Type[BaseModel] = None,
        handoff_description: str = None,
        input_guardrails: List[InputGuardrail[TContext]] = None
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.output_type = output_type
        self.handoff_description = handoff_description
        self.input_guardrails = input_guardrails or []
        
        # Configure Gemini model
        self.gemini_model = genai.GenerativeModel(
            model_name=model.model_name,  # Use model name from settings
            generation_config={
                "temperature": model.temperature,
                "top_p": model.top_p,
                "top_k": model.top_k,
                "max_output_tokens": model.max_tokens,
            }
        )

    async def process(self, query: str, context: Optional[TContext] = None) -> Any:
        try:
            # Run input guardrails if context is provided
            if context and self.input_guardrails:
                context_wrapper = RunContextWrapper(context=context)
                for guardrail in self.input_guardrails:
                    result = await guardrail.guardrail_function(context, self, query)
                    if result.tripwire_triggered:
                        raise InputGuardrailTripwireTriggered()

            # Get schema information if output_type is specified
            schema_info = ""
            if self.output_type:
                schema_info = f"""
Required JSON Schema:
{self.output_type.schema_json(indent=2)}

Your response MUST be a valid JSON object matching this schema exactly. Do not include any text outside the JSON object.
All fields are required. Ensure all values match their specified types:
- destination: string
- duration_days: integer
- budget: number (float)
- activities: array of strings
- notes: string

Example format:
{{
    "destination": "Murree Hills",
    "duration_days": 3,
    "budget": 20000.0,
    "activities": ["Visit Mall Road", "Explore Pindi Point"],
    "notes": "Weather is expected to be rainy with temperatures around 0-5°C"
}}
"""

            # Prepare the context with tools and handoffs
            prompt = f"""You are {self.name}.

Instructions:
{self.instructions}

Available tools:
{self._format_tools()}

Available handoffs:
{self._format_handoffs()}

User query: {query}

{schema_info}

IMPORTANT: You MUST respond with ONLY a valid JSON object matching the required schema. Do not include any explanatory text or markdown formatting.
Your response should be a complete JSON object with all required fields.

If you need to call a tool, include it in your response as a tool call. For example:
{{
    "tool_call": {{
        "name": "get_weather_forecast",
        "args": {{
            "city": "Murree",
            "date": "2024-03-20"
        }}
    }}
}}

After receiving the tool's response, include it in your final JSON response in the notes field.
"""

            # Generate response using Gemini
            try:
                response = await self.gemini_model.generate_content_async(prompt)
                
                if not response or not hasattr(response, 'text'):
                    raise ValueError("Empty or invalid response from Gemini API")
                
                # Process the response
                if self.output_type:
                    try:
                        response_text = response.text.strip()
                        print(f"Raw response text: {response_text}")

                        # Try to extract clean JSON from response
                        if "```json" in response_text:
                            json_str = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            json_str = response_text.split("```")[1].split("```")[0].strip()
                        else:
                            # Try to extract JSON between curly braces
                            start = response_text.find('{')
                            end = response_text.rfind('}') + 1
                            if start >= 0 and end > start:
                                json_str = response_text[start:end]
                            else:
                                json_str = response_text

                        # Clean up the JSON string and ensure proper escaping of quotes
                        json_str = json_str.strip()
                        
                        # First try to parse as is
                        try:
                            parsed_json = json.loads(json_str)
                        except json.JSONDecodeError:
                            try:
                                # If that fails, try to fix common JSON formatting issues
                                json_str = json_str.replace('\n', ' ').replace('\r', '')
                                json_str = json_str.replace('"{', '{').replace('}"', '}')
                                json_str = json_str.replace('",', ',').replace('":', ':')
                                json_str = json_str.replace('  ', ' ')
                                # Try to parse the cleaned JSON
                                parsed_json = json.loads(json_str)
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON: {str(e)}")
                                # If all parsing attempts fail, create a default response
                                parsed_json = {
                                    "destination": "Unknown",
                                    "duration_days": 1,
                                    "budget": 0.0,
                                    "activities": ["No activities available"],
                                    "notes": "Error parsing response"
                                }

                        # Check for tool calls
                        if "tool_call" in parsed_json:
                            tool_call = parsed_json["tool_call"]
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            # Call the tool
                            tool_result = await self._call_tool(tool_name, **tool_args)
                            
                            # Generate a new response with the tool result
                            new_prompt = f"""You are {self.name}.

Previous response: {json.dumps(parsed_json)}
Tool result: {tool_result}

Please update your response to include the tool result in the notes field.
You MUST respond with ONLY a valid JSON object matching the required schema.
"""
                            response = await self.gemini_model.generate_content_async(new_prompt)
                            response_text = response.text.strip()
                            
                            # Parse the updated response
                            if "```json" in response_text:
                                json_str = response_text.split("```json")[1].split("```")[0].strip()
                            elif "```" in response_text:
                                json_str = response_text.split("```")[1].split("```")[0].strip()
                            else:
                                start = response_text.find('{')
                                end = response_text.rfind('}') + 1
                                if start >= 0 and end > start:
                                    json_str = response_text[start:end]
                                else:
                                    json_str = response_text
                            
                            try:
                                parsed_json = json.loads(json_str)
                            except json.JSONDecodeError:
                                parsed_json = {
                                    "destination": "Unknown",
                                    "duration_days": 1,
                                    "budget": 0.0,
                                    "activities": ["No activities available"],
                                    "notes": f"Error parsing response with tool result: {tool_result}"
                                }

                        try:
                            # Convert to Pydantic model and validate
                            result = self.output_type(**parsed_json)
                            return result
                        except Exception as e:
                            print(f"Error creating Pydantic model: {str(e)}")
                            return self.output_type(
                                destination="Unknown",
                                duration_days=1,
                                budget=0.0,
                                activities=["No activities available"],
                                notes="Error creating response"
                            )
                    except Exception as e:
                        print(f"Error processing response: {str(e)}")
                        return self.output_type(
                            destination="Unknown",
                            duration_days=1,
                            budget=0.0,
                            activities=["No activities available"],
                            notes=str(e)
                        )
                else:
                    return response.text
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                if self.output_type:
                    return self.output_type(
                        destination="Unknown",
                        duration_days=1,
                        budget=0.0,
                        activities=["No activities available"],
                        notes=str(e)
                    )
                return str(e)
        except InputGuardrailTripwireTriggered:
            raise
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            if self.output_type:
                return self.output_type(
                    destination="Unknown",
                    duration_days=1,
                    budget=0.0,
                    activities=["No activities available"],
                    notes=error_msg
                )
            return error_msg

    def _format_tools(self) -> str:
        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools:
            if hasattr(tool, 'is_tool'):
                tool_descriptions.append(f"- {tool.__name__}: {tool.__doc__}")
        return "\n".join(tool_descriptions)

    async def _call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a tool by name with the given arguments."""
        for tool in self.tools:
            if hasattr(tool, 'is_tool') and tool.__name__ == tool_name:
                try:
                    if asyncio.iscoroutinefunction(tool):
                        return await tool(**kwargs)
                    else:
                        return tool(**kwargs)
                except Exception as e:
                    return f"Error calling tool {tool_name}: {str(e)}"
        return f"Tool {tool_name} not found."

    def _format_handoffs(self) -> str:
        if not self.handoffs:
            return "No handoffs available."
        
        handoff_descriptions = []
        for agent in self.handoffs:
            if agent.handoff_description:
                handoff_descriptions.append(f"- {agent.name}: {agent.handoff_description}")
            else:
                handoff_descriptions.append(f"- {agent.name}: {agent.instructions}")
        return "\n".join(handoff_descriptions)

class Runner:
    @staticmethod
    async def run(agent: Agent[TContext], query: str, context: Optional[TContext] = None) -> RunResult:
        final_output = None
        raw_responses = []
        try:
            final_output = await agent.process(query, context)
            raw_responses.append(str(final_output)) # Store raw response
            return RunResult(
                final_output=final_output,
                raw_responses=raw_responses,
                new_items=[final_output],
                input_guardrail_results=[],
                output_guardrail_results=[]
            )
        except InputGuardrailTripwireTriggered:
            # Handle tripwire if necessary, maybe log it
            return RunResult(
                final_output=None,
                raw_responses=raw_responses, # Include any raw responses captured before tripwire
                new_items=[],
                input_guardrail_results=[], # Potentially add tripwire info here
                output_guardrail_results=[]
            )
        # finally:
        #     # Attempt to clean up the Gemini model's resources after processing
        #     # NOTE: Commented out as potential cause of grpcio shutdown error
        #     if hasattr(agent, 'gemini_model'):
        #         # Check for a standard 'close' method first
        #         close_method = getattr(agent.gemini_model, 'close', None)
        #         # Fallback check for potential private methods if 'close' isn't found
        #         if not close_method:
        #              close_method = getattr(agent.gemini_model, '_close', None) # Example private method check
        #
        #         if close_method and callable(close_method):
        #             try:
        #                 if asyncio.iscoroutinefunction(close_method):
        #                     print("Attempting async cleanup on gemini_model...") # Debug print
        #                     await close_method()
        #                     print("Async cleanup completed.") # Debug print
        #                 else:
        #                     print("Attempting sync cleanup on gemini_model...") # Debug print
        #                     close_method()
        #                     print("Sync cleanup completed.") # Debug print
        #             except Exception as e:
        #                 print(f"Error during gemini_model cleanup: {e}") # Log error if cleanup fails
        #         else:
        #             print("No suitable close/cleanup method found on gemini_model.") # Debug print
