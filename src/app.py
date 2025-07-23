from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict

app = FastAPI()

class HouseData(BaseModel):
    area: float = Field(..., ge=0.0, description="Area of the house in square units")
    bedrooms: float = Field(..., ge=0.0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0.0, description="Number of bathrooms")
    stories: float = Field(..., ge=0.0, description="Number of stories")
    mainroad: float = Field(..., ge=0.0, le=1.0, description="Binary: 1 if on main road, 0 otherwise")
    guestroom: float = Field(..., ge=0.0, le=1.0, description="Binary: 1 if guestroom, 0 otherwise")
    basement: float = Field(..., ge=0.0, le=1.0, description="Binary: 1 if basement, 0 otherwise")
    hotwaterheating: float = Field(..., ge=0.0, le=1.0, description="Binary: 1 if hot water heating, 0 otherwise")
    airconditioning: float = Field(..., ge=0.0, le=1.0, description="Binary: 1 if air conditioning, 0 otherwise")
    parking: float = Field(..., ge=0.0, description="Number of parking spaces")
    prefarea: float = Field(..., ge=0.0, le=1.0, description="Binary: 1 if preferred area, 0 otherwise")
    furnishingstatus: float = Field(..., ge=0.0, le=2.0, description="Furnishing status: 0 (unfurnished), 1 (semi-furnished), 2 (furnished)")

@app.post("/predict")
def get_prediction(data: HouseData):
    try:
        data_dict = data.model_dump()
        result = predict(data_dict)
        return {"predicted_price": result}
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail={"error": "Model file not found"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": f"Prediction failed: {str(e)}"})