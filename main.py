from fastapi import FastAPI
from models import user_info,userInfoRequest,queryRequest
from mongoengine import connect
from pymongo import MongoClient
import torch
from langchain.llms.base import LLM
from transformers import pipeline
from llama_index import Document
from FlanT5_with_drive import create_index
from fastapi.middleware.cors import CORSMiddleware

#generic tasks 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
connect(db="gyrohead",host="localhost",port=27017)
client = MongoClient()

# create index 
index = create_index()
query_engine = index.as_query_engine()


@app.get("/hello")
async def hello():
    return "test api response "

@app.post("/verifyUser")
async def verifyUser(reqBody : userInfoRequest):
    print(reqBody.userName)
    entries = client["gyrohead"]["user_info"]
    findByUser = entries.find_one({"username":reqBody.userName})
    usrpass = findByUser["password"]
    if(usrpass == reqBody.userPassword):
        return {
            "loginUserRes":"Success"
        }
    else:
        return{
            "loginUserRes":"Failed"
        }


@app.post("/searchqQuery")
async def searchqQuery(reqBody: queryRequest):
    response = query_engine.query(reqBody.queryStr)
    responseStr = response.response
    return {"queryResponse" : responseStr}




