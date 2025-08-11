
# Simple Navigation Recommendation System

## Setup Instructions

1. **Download the Yelp Dataset**  
   Download and extract the [Yelp academic dataset](https://business.yelp.com/data/resources/open-dataset/) JSON file into the folder:  
   `data/raw/yelp_academic_dataset_business.json`

2. **Install Dependencies**  
   Install the required Python packages listed in `requirements.txt` using:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**  
   Set your OpenAI API key in a `.env` file located in the project root with the following format:  
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
4. **Run the Application**  

   Start the application from cli:
    ```bash
      python console.py
      ``` 
   Alternatively, you can run the application as a server and send the requests via a client:

   1. Start the FastAPI server by running:  
      ```bash
      python app.py
      ```  
   2. Test the system by executing the client test script:  
      ```bash
      python test/user_test.py
      ```

   Note: in data `data/embeddings.npy`and `filtered_pois.csv` generated embeddings and filtered data are stored
   based on the city selected. If you want to use another city, you need to delete the files to let them be regenerated. They are used for speed-up.

5. **Example**

An example of a request and response with retrieved points of interest (POIs) is given below:

**Request:**  
Hi, direct me to a fine Italian restaurant with more than 4 stars.

**Response:**  
```json
{
  "response": "Ristorante Mezza Luna offers fine Italian dining with a 4.0 rating and upscale ambiance. Shall I navigate you there?",
  "retrieved_pois": [
    {
      "name": "Ristorante Mezza Luna",
      "category": "Restaurants, Italian",
      "rating": 4.0,
      "price_level": "$$$",
      "address": "901 S 8th St",
      "latitude": 39.9383377,
      "longitude": -75.1562457
    },
    {
      "name": "Villa Di Roma",
      "category": "Restaurants, Italian",
      "rating": 4.0,
      "price_level": "$$",
      "address": "936 S 9th St",
      "latitude": 39.9379512,
      "longitude": -75.158238
    },
    {
      "name": "Pat Bombino's",
      "category": "Restaurants, Italian",
      "rating": 4.0,
      "price_level": "$$",
      "address": "767 S 9th St",
      "latitude": 39.9397984,
      "longitude": -75.1575767
    }
  ]
}
```
