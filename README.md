# Digital Parbhani Chat API

A conversational AI chat API built with FastAPI and Google's Gemini AI that helps users with their queries in a natural way.

## Features

- Natural conversation using Gemini AI
- Maintains conversation history per user
- Context-aware responses
- Structured response format with profile details
- Follow-up handling
- Handles medical queries, civic issues, and professional needs

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update the Gemini API key in `config.py`:
```python
GEMINI_API_KEY = "your_api_key_here"
```

4. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoint

### POST /chat
Send a message to the chat API:
```json
{
    "message": "I have a headache",
    "user_id": "user123"
}
```

The API will return a structured response:
```json
{
    "response": "Oh, I understand how uncomfortable that can be! The weather changes often cause these issues. I know Dr. Rajesh Patil near you who's really good with this. Would you like me to book an appointment with him?",
    "profiles": [
        {
            "name": "Dr. Rajesh Patil",
            "designation": "General Physician",
            "contact_number": "9876543210",
            "specialization": "General Medicine",
            "experience": "15 years",
            "rating": 4.5
        }
    ],
    "follow_up": true,
    "follow_up_type": "appointment"
}
```

Response fields:
- `response`: The natural language response
- `profiles`: List of relevant professionals with their details
- `follow_up`: Whether this conversation needs a follow-up
- `follow_up_type`: Type of follow-up needed (appointment/task/general)

## Example Usage

1. Medical Query:
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "I have a headache", "user_id": "user123"}'
```

2. Civic Issue:
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "The roads in my area are very bad", "user_id": "user123"}'
```

3. Professional Need:
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "I need to consult a lawyer", "user_id": "user123"}' 