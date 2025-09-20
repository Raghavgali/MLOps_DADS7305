from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class CaliData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class CaliResponse(BaseModel):
    response: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=CaliResponse)
async def predict_cali(cali_features: CaliData):
    try:
        features = [[
            cali_features.MedInc,
            cali_features.HouseAge,
            cali_features.AveRooms,
            cali_features.AveBedrms,
            cali_features.Population,
            cali_features.AveOccup,
            cali_features.Latitude,
            cali_features.Longitude
        ]]

        prediction = predict_data(features)
        return CaliResponse(response=float(prediction))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    