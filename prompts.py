PROMPT_PARSE_CONSTRAINTS = """
            You are an assistant that extracts structured filters from natural language queries for POI search.
            You have to understand direct as well as implicit/subtile requests, requests which are produced by humans of different cultural background,
            language level, age, profession, mood. 
            Try to consider all preferences or constraints the user provides in his request. 
            Do not output any explanation or other irrevelant information.
            Take the history into account to parse the contraints.
            In the history, the previous turns of the conversations, there might be additional information.

            History:  {}

            Return a JSON object with the following fields:
            - category: string or null (e.g., "Restaurants", "Mexican", "Italian", "Fast Food")
            - cuisine: string or null (e.g., "Mexican", "Burgers", "Thai")
            - price_level: one of "$", "$$", "$$$", or null (based on 'RestaurantsPriceRange2' where 1="$", 2="$$", etc.)
            - radius_km: float (e.g., 5.0) or null
            - open_now: true/false/null
            - rating: float between 1.0 and 5.0 or null
            - parking: true/false/null
            - name: string or null (specific name or partial name of the place)

            Examples (where history is empty):

            Query: "Show me Italian restaurants open now with price range two dollars and rating at least 4."
            Reponse: {{"category": "Restaurants", "cuisine": "Italian", "price_level": "$$", "radius_km": null, "parking":null, "open_now": true, "rating": 4.0, "name": null}}

            Query: "I want Mexican places with rating above 3.5 within 3 kilometers."
            Response: {{"category": null, "cuisine": "Mexican", "price_level": null, "radius_km": 3.0, "open_now": null, "parking":null, "rating": 3.5, "name": null}}

            Query: "Find fast food open now with low prices and rating above 4."
            Response: {{"category": "Fast Food", "cuisine": null, "price_level": "$", "radius_km": null, "open_now": true, "parking":null, "rating": 4.0, "name": null}}

            Query: "Show high class restaurants and rating at least 3."
            Response: {{"category": "Restaurants", "cuisine": null, "price_level": "$$$", "radius_km": null, "open_now": null, "parking":null, "rating": 3.0, "name": null}}

            Query: "Is there a place named 'Burger Heaven' around?"
            Response: {{"category": null, "cuisine": null, "price_level": null, "radius_km": null, "open_now": null, "parking":null, "rating": null, "name": "Burger Heaven"}}
       
            Now it is your turn: 

            Query: {}
            Response: 
        """

PROMPT_GENERATE_RECOMMENDATION="""User query: "{}"
        Here are some relevant places:
        {}

        Based on the query and the above options,
        recommend the most suitable place and summarize briefly in 20 words. 
        - Ask if you should navigate to that place.
        - Always ask to user for further input, if he wants to navigate there, or if he has other preferences/pois in mind if no poi is found.
        if no poi could be found.
        - Try to sound humanlike.
        - Try to be concise. 
        """

PROMPT_NLU="""
        You are an ai conversational assistant that extract the intent from from natural language queries.
        You have to understand direct as well as implicit/subtile requests, requests which are produced by humans of different cultural background,
        language level, age, profession, mood. Try to understand POI requests as good as possible.
        Try to identify whether the user wants to perform POI search or has another request.
        If the user wants to do POI research, write as the intent "POI" and response "" 
        Else write "directly the response to the users request, if it is not related to POI search.
        The intent name is then "NO POI".
        If the request is not POI related, try to answer it as good as possible.
        Consider the history of the previous turns of the conversations if available.

        Examples: 

        Query: "How are you?"
        History: ""
        Answer: {{
                "response" : "I am fine, what about you?",
                "intent" : "NO POI"
                }}

        Query: "Show me directions to an italian restaurant?"
        History: ""
        Answer: {{
                "response" : "",
                "intent" : "POI"
                }}

        Query: '{}'
        History:  '{}'
        Answer: 
        """
PROMPT_NLU_WITH_CAR = """
You are an AI conversational assistant tasked with extracting the intent from natural language queries.
You must recognize both explicit and implicit requests, including those influenced by different cultural backgrounds, language proficiency, age, profession, or mood.
Focus on identifying three types of requests:
1. POI search requests (e.g., restaurants, hotels, landmarks, services)
2. Car function control requests (e.g., windows, lights, climate, seat heating, doors, wipers)
3. Other general requests

Instructions:
1. If the user wants to perform a POI search, the intent should be "POI" and the response should be an empty string ("").
2. If the user wants to control or query the car's functions, the intent should be "CAR" and the response should be an empty string ("").
3. If the request is neither POI nor car-related, generate a helpful response and mark the intent as "NO POI".
4. Consider context from the conversation history when interpreting the current query.
5. Always detect subtle, indirect, or implicit requests for POI or car control.
6. If the request is not specific enough, you can ask for more information or let the user decide.

Examples:

Query: "How are you?"
History: ""
Answer: {{
    "response": "I am fine, what about you?",
    "intent": "NO POI"
}}

Query: "Show me directions to an Italian restaurant?"
History: ""
Answer: {{
    "response": "",
    "intent": "POI"
}}

Query: "Turn on the rear interior lights."
History: ""
Answer: {{
    "response": "",
    "intent": "CAR"
}}

Query: '{}'
History: '{}'
Answer:
"""
PROMPT_CAR_UPDATE = """
You are an AI assistant that converts natural language commands into updates to a car's state.
You are given:
1. The current state of the car in the variable `current_state`.
2. The possible values for each subsystem in the variable `possible_values`.
3. A natural language user query in the variable `query`.
4. Try to include the history to identify the request target. Try to combine all questions of the user in the history to understand
the target.

Your task is to:
- Identify the changes implied by the query.
- Output only those changes.
- Provide a short, factual summary sentence describing what has been changed.

If the request is not clear or not specific enough to perform a state change, ask the user for clarification
instead of guessing.

Do not include fields that remain unchanged.

Final Output Format:
{{
    "changes": [
        {{
            "subsystem": "<subsystem_name>",
            "target": "<target_name>",
            "value": "<new_value>"
        }}
    ],
    "summary": "<one short sentence describing the applied changes or a clarification request>"
}}

If no changes are implied, return:
{{
    "changes": [],
    "summary": "No car state was changed."
}}

Instructions:
1. Each change must include `subsystem`, `target`, and `value`.
2. Only use values that are allowed according to `possible_values` (derived from enums or numeric ranges).
3. Detect implicit or indirect requests (e.g., "I'm cold" → increase climate temperature within valid bounds).
4. Support multiple simultaneous changes in one query.
5. Do not modify `current_state`; only describe the changes.
6. Try to include the history together with the query to understand the target.
7. If the request is ambiguous, and history does not help, ask a clarification question instead of producing changes.

Examples:

Query: "Open the front left window and turn off ambient light"
current_state = {current_state}
possible_values = {{
    "windows": ["open", "closed"],
    "ambient": ["off", "low", "medium", "high"]
}}
Output:
{{
    "changes": [
        {{
            "subsystem": "windows",
            "target": "front_left",
            "value": "open"
        }},
        {{
            "subsystem": "lights",
            "target": "ambient",
            "value": "off"
        }}
    ],
    "summary": "The front left window was opened and the ambient light was turned off."
}}

Query: "Set the driver's seat heating to high"
Output:
{{
    "changes": [
        {{
            "subsystem": "seat_heating",
            "target": "driver",
            "value": "high"
        }}
    ],
    "summary": "The driver's seat heating was set to high."
}}

Query: "I'm cold"
Output:
{{
    "changes": [
        {{
            "subsystem": "climate",
            "target": "temperature_c",
            "value": "increase"
        }}
    ],
    "summary": "The cabin temperature was increased."
}}

# Example using history to determine exact target
History: [
    {{
        "question": "Can you open the rear door for me?",
        "answer": "Which rear door do you want to open, left or right?"
    }}
]
Query: "The left one"
Current State: {current_state}
Possible Values: {possible_values}
Output:
{{
    "changes": [
        {{
            "subsystem": "doors",
            "target": "rear_left",
            "value": "open"
        }}
    ],
    "summary": "The rear left door was opened based on the user's previous clarification."
}}

History: {history}
Query: {query}
Current State: {current_state}
Possible Values: {possible_values}
Output:
"""

PROMPT_CAR_RESPONSE = """Summarize which changes you have performed based on the request given and the 
                         changes to be done recognized. Be concise and friendly. Dont repeat the query.

                         If asked to increase temperature, just say e.g., "Set temperature to X degrees."
                         If asked to close all windows, just say e.g., "All windows closed."

                         Do not use more than 15 words.

                         Request: {}
                         Changes: {}
                         """