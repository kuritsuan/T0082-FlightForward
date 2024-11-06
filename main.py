from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import re

pattern = r'```json\s*([\s\S]*?)```'

transportation_details = ResponseSchema(name = "transportation_name", description="str form, if blank, put null.should only be filled with one of this option: uber or taxi, mtr, ferry, bus")
from_location = ResponseSchema(name="from_location", description="if it is at the start, it should be the airport. However if it is not, it should refer to the previous 'to_lcoation'")
to_location = ResponseSchema(name="to_location", description="str form, if blank, put null. the lcoation where the user want to go next from 'from_location'")

response_schema = [transportation_details, from_location, to_location]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schema)
format_instructions = output_parser.get_format_instructions()

opt= """
{
    "transportation_name": null,
    "from_location": null,
    "to_location": null
}
"""

template1 = """
This is some context about your role as a chatbot: {context}

Do not show the json structure. This is a chatbot with the user that do not know anything how does this model work

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

You can ask by doing this:
"

We’re excited to help you get to your destination! How would you like to travel to [location]?

Here are some transportation options to consider, please select your preferred transportation method:

Uber
Ferry
Bus
MTR

"

3. Check the current data that needs to be filled out, if any of this is still null, ask questions more regarding on this
4. ask if they want to go anywhere else. If yes:repeat number 1, if no:finish

Here is the conversation history: {history}

This is your current data: {opt}

Question: {question}

Answer:
"""

template2 = """
This is some context: You don't act as a chatbot, however you act as a robot that construct some formated inputs in json. 

If this is a start, make from_location: 'airport'
If any of this indicators are still blank, put in null! not a string 'null' but the null keyword

Here is the conversation history: {history}

This is your previous data: {opt}

{format_instructions}

"""

model = OllamaLLM(model="llama3")
prompt1 = ChatPromptTemplate.from_template(template1)
chain1 = prompt1 | model
prompt2 = ChatPromptTemplate.from_template(template2)
chain2 = prompt2 | model

history = ""

context = """
This is an automated input
You are a guide for the user that just recently arrive in hong kong. Ask them politely where will they stay and how will they commute.
"""


def start():
    global history
    global context
    global chain1
    global format_instructions
    global opt
    global pattern
    result = chain1.invoke({"context":context, "history": history, "question": "", "opt": opt})
    print(f"Bot: {result}")
    history += f"\nAI: {result}"
    return result


def handle_conversation(user_input):
    global history
    global context
    global chain1
    global chain2
    global format_instructions
    global opt
    global pattern
    history += f"\nUser: {user_input}"

    #inputnto program
    opt = chain2.invoke({"format_instructions":format_instructions, "history": history, "opt": opt}) 

    print("-----------------------------------------")
    # match = re.search(pattern, opt)
    # opt = str(match.group(1).strip())
    # print(opt)
    print(opt)
    print("-----------------------------------------")


    result = chain1.invoke({"context":context, "history": history, "question": user_input, "opt": opt})
    print(f"Bot: {result}")
    history += f"\nAI: {result}"
    return result



#     history = """
# bot: Welcome to Hong Kong! I'm excited to help you navigate this amazing city. Since you've just arrived, I'd like to start by asking where you plan to stay during your visit? Would you prefer a hotel in a specific area or are you open to recommendations?
# user: tsim sha tsui
# """
#     opt = chain2.invoke({"history": history}) 
#     print(opt)

start()

while True:
    user_input = input("You: ")
    handle_conversation(user_input)
