from fastapi import FastAPI ,Response 
import uvicorn

import model_load

app = FastAPI()

from transformers import AutoModelForCausalLM,AutoTokenizer

@app.get('/')
async def welcome():
    return {'message':'server checking'}

@app.get('/genai_response')
async def llm_response(text:str):
    result = model_load.get_response(text)
    return {'model_response':result}

if __name__ == '__main__':
    uvicorn.run()
    
