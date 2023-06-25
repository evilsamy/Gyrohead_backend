from mongoengine import Document, StringField
from pydantic import BaseModel

class user_info(Document):
    username = StringField()
    password = StringField()
    userEmailId = StringField()

class userInfoRequest(BaseModel):
    userName:str
    userPassword:str 

class queryRequest(BaseModel):
    queryStr:str

