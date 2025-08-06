
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
   1. Start the FastAPI server by running:  
      ```bash
      python app.py
      ```  
   2. Test the system by executing the client test script:  
      ```bash
      python test/user_test.py
      ```
