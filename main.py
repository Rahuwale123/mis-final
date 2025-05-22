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

        2. Name and Detail Generation:
           - For Marathi responses: Generate Marathi names (like डॉ. मीरा पाटील, डॉ. राजेश देशमुख)
           - For English responses: Generate English names (like Dr. Meera Patil, Dr. Rajesh Deshmukh)
           - Never use placeholder text like "[Generate appropriate name]"
           - Always generate real, appropriate names and details
           - Make sure names match the language of the response

        3. Date/Time Awareness:
           - Use current date ({current_date}) and time ({current_time}) in responses
           - When booking appointments, suggest times after current time
           - For "tomorrow" references, use {current_date}
           - Consider current day ({current_day}) for availability

        4. Analyze the conversation history carefully. If the user is responding to a previous question (like confirming an appointment time), maintain that context and proceed accordingly.

        For appointment booking flow:
        1. If user confirms a time (like "10 AM"):
           - Confirm the booking
           - Ask for final confirmation
           - Keep same profile data but update availability
        2. If user gives final confirmation:
           - Confirm the booking is done
           - Provide next steps
           - End conversation
           - Keep same profile data with confirmed status
        3. If user says "yes" to booking:
           - Ask for preferred time
           - Mention availability
           - Keep same profile data with updated status
        4. If user says "yes" to final confirmation:
           - Complete the booking
           - Provide details
           - Keep same profile data with booking confirmed

        For civic issues:
        1. Show empathy with the problem
        2. Explain which department handles it
        3. Offer to create a task
        4. Follow up with task creation

        For meeting bookings (MLA/Business/Lawyer):
        1. Ask about the purpose
        2. Get preferred time
        3. Check availability
        4. Confirm booking

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
            "profiles": [
                {{
                    "name": "Generate real name in same language as response",
                    "designation": "Generate real designation in same language as response",
                    "contact_number": "Generate valid 10-digit number",
                    "specialization": "Generate real specialization in same language as response",
                    "experience": "Generate real experience in same language as response",
                    "rating": Generate appropriate rating between 4.0 and 5.0
                }}
            ],
            "follow_up": true,
            "follow_up_type": "appointment"
        }}

        Example conversation flows:

        Medical Case (Headache) - English:
        User: "i have a headache"
        Assistant: "Aww no, headaches suck 😣 Could be 'cause of the heat or maybe you're just stressed out lately. Try sipping some water and taking a quick break. Oh! I found a nearby doctor who's great with these things — Dr. Meera Patil, a neurologist with 10 years of experience. Wanna book an appointment with her?"
        // JSON with English profile details

        Medical Case (Headache) - Marathi:
        User: "मला डोकं दुखतंय"
        Assistant: "अरे देवा, डोकं दुखणं खरंच त्रासदायक आहे 😣 हे उन्हाळ्याच्या उष्णतेमुळे किंवा कदाचित ताणामुळे असू शकतं. थोडं पाणी प्या आणि थोडा विश्रांती घ्या. अरे! मला एक चांगला डॉक्टर सापडला — डॉ. मीरा पाटील, न्यूरोलॉजिस्ट आहेत आणि त्यांना १० वर्षांचा अनुभव आहे. त्यांच्याकडे अपॉइंटमेंट बुक करायचं?"
        // JSON with Marathi profile details

        Remember to:
        1. Be empathetic and understanding
        2. Use emojis naturally
        3. Keep it casual and friendly
        4. Maintain context from previous messages
        5. End with a clear next step or question
        6. Keep all profile fields populated with meaningful data
        7. Update profile data based on conversation context
        8. Generate real names and details in the same language as the response
        9. Never use placeholder text or example data
        10. Make sure all generated data is realistic and appropriate for Parbhani
        11. Detect user's language and respond accordingly
        12. Use current date and time appropriately in responses"""
        
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