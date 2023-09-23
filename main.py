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
import openai
from ScrapApiDocToDF import crawlAndStructureAndCreateDf
from loadLocalApiDocToDF import loadLocalDocsAndStructureAndCreateDf
import pinecone
from tqdm.auto import tqdm
from IPython.display import display,Markdown
import datetime

openai.api_key = "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"
PINECONE_API_KEY = '500a4e9a-c365-4c57-a82b-b2996dde49e1'
PINECONE_API_ENV = 'asia-southeast1-gcp-free'
index_name = 'news-api-index'

#generic tasks 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# connect(db="gyrohead",host="localhost",port=27017)
# client = MongoClient()


# @app.post("/verifyUser")
# async def verifyUser(reqBody : userInfoRequest):
#     print(reqBody.userName)
#     entries = client["gyrohead"]["user_info"]
#     findByUser = entries.find_one({"username":reqBody.userName})
#     usrpass = findByUser["password"]
#     if(usrpass == reqBody.userPassword):
#         return {
#             "loginUserRes":"Success"
#         }
#     else:
#         return{
#             "loginUserRes":"Failed"
#         }

@app.get("/checkServer")
async def root():
    return {"message": "Server is up"}



# # create index 
# index = create_index()
# query_engine = index.as_query_engine()

# @app.post("/searchqQuery")
# async def searchqQuery(reqBody: queryRequest):
#     response = query_engine.query(reqBody.queryStr)
#     responseStr = response.response
#     return {"queryResponse" : responseStr}

# @app.post("/searchqQuery")
# async def searchqQuery(reqBody: queryRequest):
#     responseStr = search_for_query(reqBody.queryStr)
#     return {"queryResponse" : responseStr}

# # df = crawlAndStructureAndCreateDf()
# df = loadLocalDocsAndStructureAndCreateDf()

# def create_pinecone_index():
#     # Define index name
#     # index_name = 'news-api-index'

#     # Initialize connection to Pinecone
#     pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

#     # Check if index already exists, create it if it doesn't
#     if index_name not in pinecone.list_indexes():
#         pinecone.create_index(index_name, dimension=1536, metric='dotproduct')

# print("call create_pinecone_index")
# create_pinecone_index()
# print("pinecone index created")


# def save_api_knowledge():

#     df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

#     # Add an 'id' column to the DataFrame
#     from uuid import uuid4
#     df['id'] = [str(uuid4()) for _ in range(len(df))]

#     # Fill null values in 'title' column with 'No Title'
#     df['fname'] = df['fname'].fillna('No Title')

#     df.rename(columns={'fname': 'title'}, inplace=True)

#     # Connect to the index and view index stats
#     index = pinecone.Index(index_name)
#     # index.describe_index_stats()

#     batch_size = 100  # how many embeddings we create and insert at once

#     # Convert the DataFrame to a list of dictionaries
#     chunks = df.to_dict(orient='records')

#     # Upsert embeddings into Pinecone in batches of 100
#     for i in tqdm(range(0, len(chunks), batch_size)):
#         i_end = min(len(chunks), i+batch_size)
#         meta_batch = chunks[i:i_end]
#         ids_batch = [x['id'] for x in meta_batch]
#         embeds = [x['embeddings'] for x in meta_batch]
#         meta_batch = [{
#             'title': x['title'],
#             'text': x['text'],
#             'url': x['url']
#         } for x in meta_batch]
#         to_upsert = list(zip(ids_batch, embeds, meta_batch))
#         index.upsert(vectors=to_upsert)

# print("call save_api_knowledge")
# save_api_knowledge()
# print("saved in db sucsessfuly")

# def search_for_query(query):
#     # Connect to the index and view index stats
#     index = pinecone.Index(index_name)  

#     embed_model = "text-embedding-ada-002"
#     user_input = query

#     embed_query = openai.Embedding.create(
#         input=user_input,
#         engine=embed_model
#     )

#     # retrieve from Pinecone
#     query_embeds = embed_query['data'][0]['embedding']

#     # get relevant contexts (including the questions)
#     response = index.query(query_embeds, top_k=5, include_metadata=True)

#     print(response)
#     contexts = [item['metadata']['text'] for item in response['matches']]

#     augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+ user_input

#     # system message to assign role the model
#     system_msg = f"""You are a helpul customer support assistant assistant expert. Answer questions based on the context provided, or say I don't know.".
#     """

#     chat = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": augmented_query}
#         ]
#     )

#     # display(Markdown(chat['choices'][0]['message']['content']))
#     print(chat['choices'][0]['message']['content'])
#     return chat['choices'][0]['message']['content']


# print("querying started")
# print(datetime.datetime.now)
# search_for_query("Credit Bureau Data")
# print("query completed") 
# print(datetime.datetime.now)
