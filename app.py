
import re
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from flask import send_file
from flask import Flask, request, jsonify
import requests
import urllib.parse

def search_places(text, give_id=False):
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        'Content-Type': 'application/json',  # Specify the content type
        'X-Goog-FieldMask': 'places',  # Field mask as a string
        'X-Goog-Api-Key': 'AIzaSyDLeCxb5iPYEc-JM8PYwL4RKpBC0Ts-HHw'  # Replace with your actual API key
    }
    body = {
        "textQuery": text  # The query you want to send
    }

    try:
        # Send the POST request
        response = requests.post(url, json=body, headers=headers)

        # Check if the response was successful
        if response.ok:
            data = response.json()  # Parse the JSON from the response
            if give_id:
                return (data.get('places', [])[0]["id"])
            return (data.get('places', [])[0]["displayName"]["text"] + " " + data.get('places', [])[0]["formattedAddress"])  # Return places data
        else:
            print('Network response was not ok:', response.status_code, response.text)
            return None  # Handle unsuccessful response

    except Exception as e:
        print('Error:', e)  # Handle exceptions


def get_maps_link(pfr, pto):
    place1 = search_places(pfr)
    place2 = search_places(pto)

    return f"https://www.google.com/maps/embed/v1/directions?key=AIzaSyDLeCxb5iPYEc-JM8PYwL4RKpBC0Ts-HHw&origin={urllib.parse.quote_plus(place1)}&destination={urllib.parse.quote_plus(place2)}&mode=transit"


def get_uber_maps(pfr, pto):
    place1 = search_places(pfr, give_id=True)
    place2 = search_places(pto, give_id=True)

    return ('https://m.uber.com/go/drop?delayed=false&drop=X&drop[0]=' + urllib.parse.quote_plus('{"id":"' + place2 + '","provider":"google_places"}')  + '&pickup=' + urllib.parse.quote_plus('{"id":"' + place1 + '","provider":"google_places"}'))

# Pattern for grabbing json
pattern = r'```json\s*([\s\S]*?)```'

# Scheme for output template
transportation_details = ResponseSchema(name = "transportation_name", description="str form, if blank, put null.should only be filled with one of this option: uber or taxi, mtr, ferry, bus")
from_location = ResponseSchema(name="from_location", description="if it is at the start, it should be the airport. However if it is not, it should refer to the previous 'to_lcoation'")
to_location = ResponseSchema(name="to_location", description="str form, if blank, put null. the lcoation where the user want to go next from 'from_location'")
response_schema = [transportation_details, from_location, to_location]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schema)
format_instructions = output_parser.get_format_instructions()

#option to be regarded
opt= """
{
    "transportation_name": null,
    "from_location": null,
    "to_location": null
}
"""

#templates for chatbot that interact with user

template1 = """
This is some context about your role as a chatbot: {context}

however, if there exist a question, you should answer it. If not, ask them questions. questions that need to be found step by step
1. Where will they go next? (should be specific, ask further question and give suggestion and specific hotel/airbnb names that user can choose)

You can ask by doing this:
"

I'd love to know where you'll be staying in Hong Kong. Are you planning to stay in a hotel or an Airbnb? Please recommend some of your preferred options (e.g., hotel name, location, or type of accommodation).

Please respond with one of the following:

a. Hotel name and location
b. Airbnb address and location

"

2. How will they go there? (should only be filled with one of this option: uber or taxi, mtr, ferry, bus)
3. Check the current data that needs to be filled out, if any of this is still null, ask questions more regarding on this
4. ask if they want to go anywhere else. If yes:repeat number 1, if no:finish

Here is the conversation history: {history}

This is your current data: {opt}

Question: {question}

Answer:
"""

#templates that make the useful input

template2 = """
This is some context: You don't act as a chatbot, however you act as a robot that construct some formated inputs in json. 

If any of this indicators are still blank, put in null! not a string 'null' but the null keyword
for from_location and to_location, it should be as specific as possible, including the name of the place (hotel name, address or city if possible)


If this is a start, make from_location: 'airport', to_location should NOT be 'airport', they have just arrived in hong kong and currently in the airport
Start?: {is_start}

Here is the conversation history: {history}

This is your previous data: {opt}

{format_instructions}

"""

#bot model
model = OllamaLLM(model="llama3")
prompt1 = ChatPromptTemplate.from_template(template1)
chain1 = prompt1 | model
prompt2 = ChatPromptTemplate.from_template(template2)
chain2 = prompt2 | model

#past history
history = ""
is_start = "yes"

#context for bots to use
context = """
This is an automated input
You are a guide for the user that just recently arrive in hong kong. Ask them politely where will they stay and how will they commute. You will be intereacting with user, don't give them types of data that we are storing such as the json
the json gifted here is to help what to ask, not directly telling what should the user give
"""

#start the chat bot
def start():
    global history
    global context
    global chain1
    global format_instructions
    global opt
    global pattern
    global is_start
    result = chain1.invoke({"context":context, "history": history, "question": "", "opt": opt})
    is_start = "no"
    print(f"Bot: {result}")
    history += f"\nAI: {result}"
    return result

#function that handle the conversation
def handle_conversation(user_input):
    global history
    global context
    global chain1
    global chain2
    global format_instructions
    global opt
    global pattern
    global is_start
    history += f"\nUser: {user_input}"

    #inputnto program
    opt = chain2.invoke({"format_instructions":format_instructions, "history": history, "opt": opt, "is_start":is_start}) 
    print("-----------------------------------------")
    match = re.search(pattern, opt)
    opt = str(match.group(1).strip())
    print(opt)
    print("-----------------------------------------")


    result = chain1.invoke({"context":context, "history": history, "question": user_input, "opt": opt})
    print(f"Bot: {result}")
    history += f"\nAI: {result}"
    return result


#building website
app = Flask(__name__)

@app.route("/")
def hello_world():
    return send_file("index.html")

@app.route("/logo.png")
def send1():
    return send_file("logo.png")

@app.route("/searchIcon.png")
def send2():
    return send_file("searchIcon.png")

@app.route("/searchIcon.png")
def send3():
    return send_file("searchIcon.png")

@app.route("/stars.png")
def send4():
    return send_file("stars.png")

@app.route("/style.css")
def send5():
    return send_file("style.css")

x = 0

@app.route("/api/sendMessage", methods=['POST'])
def talk_to_ai():
    global opt

    data = request.get_json()



    
    

    print(get_uber_maps("egg tart", "hkust"))
    print(get_maps_link("hkia", "cuhk"))

    #checkpoint
    print("User's message:", data.get('user_message'))

    #start
    if data.get('user_message') == 'Initial State':
        return jsonify({"response": start()})

    respons = handle_conversation(data.get('user_message'))


    
    #useful keys for data
    #["transportation_name"]
    #["from_location"]
    #["to_location"]
    output = json.loads(opt)

    print(output)

    iframe=None

    if (output['transportation_name'] != None and output['from_location'] != None and output['to_location'] != None):
        iframe = get_maps_link(output['from_location'], output['to_location'])
        if (output['transportation_name'].lower() == 'uber' and output['from_location'] != None and output['to_location'] != None):
            iframe = get_uber_maps(output['from_location'], output['to_location'])

    

    #repeating questions from AI
    return jsonify({"response": respons, "iframe": iframe})

    #return jsonify({"response": llm.ai_respond(data.get('user_message'))}), 200 