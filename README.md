
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
   Create a huggingface account in case you want to use ollama models with tokenizers for huggingface.
   Provide the token here:

   ```
   HF_TOKEN=your_hf_token
   ```
   On the HF website ask for access for each model you want to use. Otherwise you will get a 401 error.

4. **Run the Application**  

   Start the application from cli:
    ```bash
      python console.py
      ``` 
   Alternative you can run the application as a server and send the requests via a client:

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
