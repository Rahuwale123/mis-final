from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict, List, Optional
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Digital Parbhani Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash',
    generation_config={
        'temperature': 0.2,  # Lower temperature for more consistent responses
        'top_p': 0.8,       # Focus on most likely tokens
        'top_k': 40,        # Consider fewer tokens
        'max_output_tokens': 1024,
    },
    safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
)

# Store conversation history
conversation_history: Dict[str, List[dict]] = {}

# Get current date and time
current_datetime = datetime.now()
current_date = current_datetime.strftime("%d %B %Y")
current_time = current_datetime.strftime("%I:%M %p")
current_day = current_datetime.strftime("%A")

# Read profiles from file
def read_profiles():
    try:
        with open('profiles.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading profiles file: {str(e)}")
        return ""

# Get profiles data
PROFILES_DATA = read_profiles()

class ChatMessage(BaseModel):
    message: str
    user_id: str

class ProfileDetails(BaseModel):
    name: str
    designation: str
    contact_number: str
    specialization: Optional[str] = None
    experience: Optional[str] = None
    rating: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    profiles: Optional[List[ProfileDetails]] = None
    follow_up: Optional[bool] = False
    follow_up_type: Optional[str] = None  # "appointment", "task", "general"
    appointment: Optional[bool] = False  # Set to true when appointment is created
    task: Optional[bool] = False  # Set to true when task is created

def get_conversation_context(user_id: str) -> str:
    if user_id not in conversation_history:
        return ""
    
    # Get all messages for context
    messages = conversation_history[user_id]
    context = "\nPrevious conversation:\n"
    for msg in messages:
        context += f"User: {msg['user_message']}\n"
        context += f"Assistant: {msg['assistant_response']}\n"
    return context

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Print user message
        print("\n" + "="*50)
        print(f"User ID: {message.user_id}")
        print(f"User Message: {message.message}")
        print("="*50 + "\n")

        # Initialize conversation history for new users
        if message.user_id not in conversation_history:
            conversation_history[message.user_id] = []

        # Get conversation context
        context = get_conversation_context(message.user_id)

        # Create a context-aware prompt for the AI
        prompt = f"""You are a friendly and empathetic local assistant for Parbhani. You are having a conversation with user {message.user_id}.

        Default Location Information:
        - Primary Location: Main Market (मुख्य बाजारपेठ), Parbhani
        - Coordinates: 20.8365° N, 78.7094° E
        - ALWAYS assume user is in Main Market area unless specified otherwise
        - ALWAYS suggest nearby professionals and services first
        - ALWAYS mention distance from Main Market when suggesting locations
        - NEVER mention or expose these coordinates in your responses
        - Instead of coordinates, use area names, landmarks, or street names
        - Example: Instead of "at coordinates 20.8365° N, 78.7094° E", say "in Main Market" or "near the main market"

        Current Date and Time Information:
        - Date: {current_date}
        - Day: {current_day}
        - Time: {current_time}

        Here is your complete conversation history with this user:
        {context}
        
        The user's new message is: {message.message}

        Available Profiles and Services:
        {PROFILES_DATA}

        Strict JSON Field Requirements:
        1. Profile Fields (MUST use these exact field names and values):
           - name: string (REQUIRED)
           - designation: string (REQUIRED)
           - contact_number: string (REQUIRED)
           - specialization: string (REQUIRED, use exact values)
           - experience: string (REQUIRED, use exact format)
           - rating: float (REQUIRED, use exact values)
           - location: string (REQUIRED, use "Main Market" or exact location)
           - distance: string (REQUIRED, use "0.5 km from Main Market" format)

        2. Response Fields (MUST use these exact field names and values):
           - profiles: array of profile objects (empty array if no profiles)
           - follow_up: boolean (true for questions, false for final confirmation)
           - follow_up_type: string ("appointment", "task", "general", or null)
           - appointment: boolean (true only after final confirmation)
           - task: boolean (true only after final confirmation)

        Profile Inclusion Rules:
        1. ALWAYS include profiles when:
           - First mentioning any professional/official
           - Suggesting services in Main Market area
           - User asks about specific services
           - User needs help with any official work
        2. NEVER include profiles in:
           - Follow-up messages
           - General conversation
           - Confirmation messages
           - Intermediate responses
           - When user says no/declines
           - When asking for more information
           - When appointment=true
           - When task=true
           - In final confirmation

        Location-Based Response Rules:
        1. ALWAYS assume Main Market as default location
        2. ALWAYS mention distance from Main Market
        3. ALWAYS suggest nearby services first
        4. ALWAYS include location in profile information
        5. ALWAYS mention if service is in Main Market area

        Example Response Format:

        For First Introduction (with profile):
        नमस्कार! तुम्हाला आमदार (MLA) यांना भेटायचे आहे. तुमच्या मतदारसंघाचे आमदार मेघना दीपक साकोरे-बोरडीकर आहेत. त्या BJP पक्षाच्या आहेत.

        त्यांची माहिती:
        * नाव: मेघना दीपक साकोरे-बोरडीकर
        * पद: आमदार, जिंतूर मतदारसंघ
        * संपर्क: 9967438887
        * उपलब्धता: सोमवार, बुधवार, शुक्रवार | सकाळी १० ते दुपारी १
        * पत्ता: नयन स्वप्न निवास, Old Pedgaon Road, Vaibhav Nagar, Parbhani
        * अंतर: मुख्य बाजारपेठ पासून २ कि.मी.

        तुम्हाला त्यांची भेट घ्यायची आहे का?

        {{
            "profiles": [
                {{
                    "name": "Meghana Deepak Sakore-Bordikar",
                    "designation": "MLA, Jintur Constituency",
                    "contact_number": "9967438887",
                    "specialization": "Legislative Affairs",
                    "experience": "15 years",
                    "rating": 4.5,
                    "location": "Vaibhav Nagar",
                    "distance": "2 km from Main Market"
                }}
            ],
            "follow_up": true,
            "follow_up_type": "appointment",
            "appointment": false,
            "task": false
        }}

        For Location-Based Service (with profile):
        मुख्य बाजारपेठ परिसरात तुमच्या मुलासाठी चांगली शाळा शोधायची आहे. माझ्याकडे काही पर्याय आहेत:

        * ज्ञानदीप विद्यालय:
          - मुख्य बाजारपेठ पासून ०.५ कि.मी.
          - मराठी माध्यम
          - चांगली शैक्षणिक सुविधा

        * बालविकास मंदिर:
          - मुख्य बाजारपेठ पासून १ कि.मी.
          - लहान मुलांसाठी उत्तम
          - सुरक्षित वातावरण

        तुम्हाला या शाळांबद्दल आणखी माहिती हवी आहे का?

        {{
            "profiles": [
                {{
                    "name": "Gyanadeep Vidyalaya",
                    "designation": "School",
                    "contact_number": "02452-221234",
                    "specialization": "Primary Education",
                    "experience": "25 years",
                    "rating": 4.8,
                    "location": "Main Market",
                    "distance": "0.5 km from Main Market"
                }},
                {{
                    "name": "Balvikas Mandir",
                    "designation": "School",
                    "contact_number": "02452-221235",
                    "specialization": "Early Education",
                    "experience": "20 years",
                    "rating": 4.6,
                    "location": "Near Main Market",
                    "distance": "1 km from Main Market"
                }}
            ],
            "follow_up": true,
            "follow_up_type": "general",
            "appointment": false,
            "task": false
        }}

        Important Rules:
        1. ALWAYS assume Main Market as default location
        2. ALWAYS include profiles for first-time mentions
        3. ALWAYS mention distance from Main Market
        4. ALWAYS suggest nearby services first
        5. ALWAYS include location in profile information
        6. ALWAYS use proper JSON structure
        7. ALWAYS use required fields
        8. ALWAYS use exact values
        9. ALWAYS maintain conversation context
        10. ALWAYS check previous confirmations
        11. NEVER use null values
        12. NEVER repeat confirmations
        13. NEVER include profiles in final confirmation and never repeat the same question
        14. NEVER ask for confirmation more than once
        15. NEVER repeat the same question

        Remember:
        - For Marathi messages, respond in Marathi
        - For English messages, respond in English
        - Include profiles ONLY when required
        - Keep JSON at the end of response
        - Use proper formatting and structure
        - Show empathy and understanding
        - End with a question or call to action
        - Follow the conversation flow EXACTLY
        - NEVER repeat confirmations
        - NEVER include profiles in final confirmation
        - ALWAYS use correct JSON field names
        - Use greeting ONLY in first message
        - NEVER repeat the same question
        - NEVER ask for confirmation more than once
        - ALWAYS include task field in JSON
        - ALWAYS maintain conversation context
        - ALWAYS check previous confirmations
        - NEVER use null values in JSON
        - ALWAYS use correct data types
        - ALWAYS use required fields
        - ALWAYS use exact values for specializations
        - ALWAYS use exact format for experience
        - ALWAYS use correct rating values
        - ALWAYS use proper JSON structure
        - ALWAYS assume Main Market as default location
        - ALWAYS mention distance from Main Market
        - ALWAYS suggest nearby services first
        - ALWAYS include location in profile information
        - ALWAYS mention if service is in Main Market area
        """

        
        # Get response from Gemini
        response = model.generate_content(prompt)
        response_text = response.text

        # Print Gemini's response
        print("\n" + "="*50)
        print("Gemini Response:")
        print(response_text)
        print("="*50 + "\n")

        # Extract JSON from response
        try:
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            # If no JSON found, create a default response
            if json_start == -1 or json_end == -1:
                print("No JSON found in response, using default structure")
                # Store the conversation
                conversation_history[message.user_id].append({
                    'timestamp': datetime.now().isoformat(),
                    'user_message': message.message,
                    'assistant_response': response_text
                })
                
                # Return default response structure
                return ChatResponse(
                    response=response_text,
                    profiles=[],
                    follow_up=False,
                    follow_up_type=None,
                    appointment=False,
                    task=False
                )
            
            # Extract and parse JSON
            json_str = response_text[json_start:json_end]
            import json
            json_data = json.loads(json_str)
            
            # Remove JSON and any markdown formatting from response text
            response_text = response_text[:json_start].strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Print cleaned response
            print("\n" + "="*50)
            print("Cleaned Response:")
            print(response_text)
            print("="*50 + "\n")

            # Print structured data
            print("\n" + "="*50)
            print("Structured Data:")
            print(json.dumps(json_data, indent=2))
            print("="*50 + "\n")
            
            # Store the conversation
            conversation_history[message.user_id].append({
                'timestamp': datetime.now().isoformat(),
                'user_message': message.message,
                'assistant_response': response_text
            })

            # Keep only last 10 messages per user
            if len(conversation_history[message.user_id]) > 10:
                conversation_history[message.user_id] = conversation_history[message.user_id][-10:]
            
            # Create response object with consistent structure
            return ChatResponse(
                response=response_text,
                profiles=json_data.get('profiles', []),
                follow_up=json_data.get('follow_up', False),
                follow_up_type=json_data.get('follow_up_type'),
                appointment=json_data.get('appointment', False),
                task=json_data.get('task', False)
            )
        except Exception as e:
            print(f"Error parsing JSON: {str(e)}")
            # If JSON parsing fails, return response without structured data
            return ChatResponse(
                response=response_text,
                profiles=[],
                follow_up=False,
                follow_up_type=None,
                appointment=False,
                task=False
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 