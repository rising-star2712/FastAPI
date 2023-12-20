import json
from fastapi import FastAPI , File , UploadFile , HTTPException
from roboflow import Roboflow
from fastapi.responses import HTMLResponse  
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

rf = Roboflow(api_key="EVotXOXaH5DqQ4x6PZH8")
project = rf.workspace().project("inventory-of-steel-rods-um0a3")
model = project.version(12).model

project1 = rf.workspace().project("ssebowabrick")
model1 = project1.version(1).model
class PredictionInput(BaseModel):
    img : str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the frontend domain(s) you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/')
async def predict(input_data: PredictionInput):
  try:  
    #with open(file.filename, "wb") as f:
    #    f.write(file.file.read())
    image_url = input_data.img
    
    prediction_json = model1.predict(image_url , hosted = True , confidence=20 , overlap=30 ).json()
   
    prediction_list = prediction_json.get("predictions" , [])

    #total_8 = 0
    #total_rods = 0
    #total_16 = 0
    #total_32 = 0
    total_brick = 0

    for i in prediction_list:
       if i.get("class") == "brick":
          total_brick += 1

    #for prediction in prediction_list:
    # Check if the class is "8 mm"
    #  if prediction.get("class") == "8 mm":
    #     total_8 += 1
    #  if prediction.get("class") == "16 mm":
    #     total_16 += 1
    #  if prediction.get("class") == "32 mm":
    #     total_32 += 1

    #total_rods = total_8 + total_32 + total_16
    
    
    return{ "bricks" : total_brick }
  except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  