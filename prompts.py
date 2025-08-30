PROMPT_PARSE_CONSTRAINTS = """
            You are an assistant that extracts structured filters from natural language queries for POI search.
            You have to understand direct as well as implicit/subtile requests, requests which are produced by humans of different cultural background,
            language level, age, profession, mood. 
            Try to consider all preferences or constraints the user provides in his request. 
            Do not output any explanation or other irrevelant information.
            Take the history into account to parse the contraints.
            In the history, the previous turns of the conversations, there might be additional information.

            Query: {}

            History:  {}

            Return a JSON object with the following fields:
            - category: string or null (e.g., "Restaurants", "Mexican", "Italian", "Fast Food")
            - cuisine: string or null (e.g., "Mexican", "Burgers", "Thai")
            - price_level: one of "$", "$$", "$$$", or null (based on 'RestaurantsPriceRange2' where 1="$", 2="$$", etc.)
            - radius_km: float (e.g., 5.0) or null
            - open_now: true/false/null
            - rating: float between 1.0 and 5.0 or null
            - name: string or null (specific name or partial name of the place)

            Examples (where history is empty):

            Query: "Show me Italian restaurants open now with price range two dollars and rating at least 4."
            {{"category": "Restaurants", "cuisine": "Italian", "price_level": "$$", "radius_km": null, "open_now": true, "rating": 4.0, "name": null}}

            Query: "I want Mexican places with rating above 3.5 within 3 kilometers."
            {{"category": null, "cuisine": "Mexican", "price_level": null, "radius_km": 3.0, "open_now": null, "rating": 3.5, "name": null}}

            Query: "Find fast food open now with low prices and rating above 4."
            {{"category": "Fast Food", "cuisine": null, "price_level": "$", "radius_km": null, "open_now": true, "rating": 4.0, "name": null}}

            Query: "Show high class restaurants and rating at least 3."
            {{"category": "Restaurants", "cuisine": null, "price_level": "$$$", "radius_km": null, "open_now": null, "rating": 3.0, "name": null}}

            Query: "Is there a place named 'Burger Heaven' around?"
            {{"category": null, "cuisine": null, "price_level": null, "radius_km": null, "open_now": null, "rating": null, "name": "Burger Heaven"}}
        """

PROMPT_GENERATE_RECOMMENDATION="""User query: "{}"
        Here are some relevant places:
        {}

        Based on the query and the above options,
        recommend the most suitable place and summarize briefly in 20 words. 
        Ask if you should navigate to that place. Always ask to user for further input, if he wants to navigate there, or if he has other preferences/pois in mind
        if no poi could be found.
        Try to sound humanlike.
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