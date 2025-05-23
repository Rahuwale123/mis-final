from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict, List, Optional
from datetime import datetime

app = FastAPI(title="Digital Parbhani Chat API")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Store conversation history
conversation_history: Dict[str, List[dict]] = {}

# Get current date and time
current_datetime = datetime.now()
current_date = current_datetime.strftime("%d %B %Y")
current_time = current_datetime.strftime("%I:%M %p")
current_day = current_datetime.strftime("%A")

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

def get_conversation_context(user_id: str) -> str:
    if user_id not in conversation_history:
        return ""
    
    # Get last 5 messages for context
    recent_messages = conversation_history[user_id][-5:]
    context = "\nPrevious conversation:\n"
    for msg in recent_messages:
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

        Current Date and Time Information:
        - Date: {current_date}
        - Day: {current_day}
        - Time: {current_time}

        Here is your previous conversation with this user:
        {context}
        
        The user's new message is: {message.message}

        Important Instructions:
        1. Language Detection and Response:
           - First, detect if the user's message is in Marathi or English
           - If the message contains Marathi characters (like म, न, य, etc.), respond in Marathi
           - If the message is in English, respond in English
           - Never mix languages in the same response
           - Keep the friendly tone in both languages

        2. Profile Information Rules:
           - ONLY include profile information when:
             * First suggesting a specific professional (doctor, lawyer, etc.)
             * First mentioning a municipal official or department head
             * User explicitly asks for someone's details
           - DO NOT include profile information for:
             * General greetings (hi, hello, etc.)
             * General questions about services
             * Weather updates
             * General information requests
             * Casual conversation
             * Follow-up messages about the same professional
           - When profiles are needed, ALWAYS include ALL these fields:
             * name: Full name of the professional
             * designation: Their professional title
             * contact_number: A valid 10-digit phone number
             * specialization: Their specific area of expertise
             * experience: Years of experience
             * rating: A rating between 4.0 and 5.0
           - Make sure all information is realistic and appropriate for Parbhani
           - Never use placeholder text or example data

        3. Conversation Flow and Context:
           - Analyze the conversation history carefully to understand the current state
           - For appointment booking, follow this exact flow:
             1. Initial Request (e.g., "I have a headache"):
                * Show empathy
                * Suggest a professional
                * Ask if they want to book
                * Include complete profile (ONLY at this first mention)
                * Set follow_up=true, follow_up_type="appointment"
             
             2. User Accepts Booking ("yes"):
                * Suggest a specific time
                * DO NOT include profile (already mentioned)
                * Set follow_up=true, follow_up_type="appointment"
             
             3. User Confirms Time ("yes" or specific time):
                * Confirm the appointment
                * DO NOT include profile (already mentioned)
                * Set follow_up=false, follow_up_type=null
                * End the booking flow
             
             4. Any Further "yes" or "confirm":
                * Acknowledge the confirmation
                * DO NOT include profile (already mentioned)
                * Set follow_up=false, follow_up_type=null
                * End the booking flow
           
           - For civic issues:
             1. Initial Report (e.g., "roads are bad"):
                * Show empathy
                * Mention relevant department/official
                * Include their profile (ONLY at first mention)
                * Ask if they want to report
                * Set follow_up=true, follow_up_type="task"
             
             2. User Accepts ("yes"):
                * Confirm the report
                * DO NOT include profile (already mentioned)
                * Set follow_up=false, follow_up_type=null
           
           - For general conversation:
             * Keep responses natural and contextual
             * Don't include profiles unless first mentioning someone
             * Set follow_up=false, follow_up_type=null

        4. Date/Time Awareness:
           - Use current date ({current_date}) and time ({current_time}) in responses
           - When booking appointments, suggest times after current time
           - For "tomorrow" references, use {current_date}
           - Consider current day ({current_day}) for availability

        Always maintain a friendly, conversational tone. Use phrases and emojis like:
        - "Aww no, that sucks 😣"
        - "Perfect 😌"
        - "Boom, it's done ✅"
        - "Gotcha! 😎"
        - "Cool! Task created ✅"

        Marathi phrases to use when user speaks in Marathi:
        - "नमस्कार, कसे आहात?" (Hello, how are you?)
        - "काय मदत हवी?" (What help do you need?)
        - "ठीक आहे, मी तुम्हाला मदत करतो" (Okay, I'll help you)
        - "चला बुक करूया" (Let's book it)
        - "झालं! ✅" (Done! ✅)

        Keep your response natural and engaging. Don't include any JSON or technical details in your response text.

        After your response, provide the following information in JSON format (but don't include the word 'json' or any markdown formatting):
        {{
            "profiles": [],  // Include profiles ONLY when first mentioning a professional or official
            "follow_up": false,  // Set to true only when expecting a specific follow-up action
            "follow_up_type": null  // Set to "appointment", "task", or "general" only when follow_up is true
        }}

        Example conversation flows:

        General Greeting - English:
        User: "hi"
        Assistant: "Hey there! 👋 How can I help you today?"
        // Empty profiles array, no follow-up needed

        Medical Case (Headache) - English:
        User: "i have a headache"
        Assistant: "Aww no, headaches suck 😣 Could be 'cause of the heat or maybe you're just stressed out lately. Try sipping some water and taking a quick break. Oh! I found a nearby doctor who's great with these things — Dr. Meera Patil, a neurologist with 10 years of experience. Wanna book an appointment with her?"
        // Include complete profile details since first mentioning the doctor:
        {{
            "profiles": [
                {{
                    "name": "Dr. Meera Patil",
                    "designation": "Neurologist",
                    "contact_number": "9876543210",
                    "specialization": "Headache and Migraine Treatment",
                    "experience": "10 years",
                    "rating": 4.8
                }}
            ],
            "follow_up": true,
            "follow_up_type": "appointment"
        }}

        Civic Issue (Bad Roads) - English:
        User: "roads are bad"
        Assistant: "Oh no, that's terrible! 😣 Bad roads are the worst. I can help you report this to Mr. Rajesh Deshmukh, the Municipal Engineer at Parbhani Municipal Corporation. He's been handling infrastructure issues for 8 years. Would you like me to report this issue?"
        // Include complete profile details since first mentioning the official:
        {{
            "profiles": [
                {{
                    "name": "Mr. Rajesh Deshmukh",
                    "designation": "Municipal Engineer",
                    "contact_number": "9876543211",
                    "specialization": "Infrastructure Management",
                    "experience": "8 years",
                    "rating": 4.5
                }}
            ],
            "follow_up": true,
            "follow_up_type": "task"
        }}

        Remember to:
        1. Be empathetic and understanding
        2. Use emojis naturally
        3. Keep it casual and friendly
        4. Maintain context from previous messages
        5. Follow the exact conversation flow for appointments and civic issues
        6. Only include profiles when first mentioning someone
        7. Always include ALL required profile fields when first suggesting a professional
        8. Detect user's language and respond accordingly
        9. Use current date and time appropriately in responses"""
        
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
            if json_start != -1 and json_end != -1:
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
                    profiles=json_data.get('profiles', [{
                        "name": "Generate real name in same language as response",
                        "designation": "Generate real designation in same language as response",
                        "contact_number": "Generate valid 10-digit number",
                        "specialization": "Generate real specialization in same language as response",
                        "experience": "Generate real experience in same language as response",
                        "rating": 4.5
                    }]),
                    follow_up=json_data.get('follow_up', False),
                    follow_up_type=json_data.get('follow_up_type')
                )
        except Exception as e:
            print(f"Error parsing JSON: {str(e)}")

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 