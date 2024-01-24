# Genai-fine-tunning-on-korean-database
This repository includes code for extracting responses from a fine-tuned model for Question & Answering (Q&A) tasks. The primary focus is on loading a Korean-based model, specifically nlpai-lab/kullm-polyglot-12.8b-v2, from the Hugging Face model hub. The model is fine-tuned on a Korean insurance dataset containing 100 sample questions and answers, and the resulting model is saved and pushed to the repository as pratik33/nlpai-lab_kullm-polyglot-12_8b-v2_custom-35.

Usage
Follow these steps to try out the code:

Clone this repository:
git clone https://github.com/your_username/GenAI-fine-tuning-on-korean-database.git
Create a virtual environment:

bash
Copy code
cd GenAI-fine-tuning-on-korean-database
python -m venv venv
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the server:

bash
Copy code
python main.py

Model Loading
The model_load.py script handles the loading of the pre-trained Korean model nlpai-lab/kullm-polyglot-12.8b-v2 from the Hugging Face model hub. The fine-tuned model on the insurance dataset is loaded from the repository as pratik33/nlpai-lab_kullm-polyglot-12_8b-v2_custom-35.

API using FastAPI
The server utilizes FastAPI to expose an API for extracting responses from the loaded model. After running main.py, the API can be accessed to obtain responses for given questions.

Feel free to explore and adapt the code according to your needs. If you encounter any issues or have suggestions, please open an issue or submit a pull request.

Happy fine-tuning and Q&A modeling with GenAI!


