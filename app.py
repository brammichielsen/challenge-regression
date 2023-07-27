import fastapi
import uvicorn
from pydantic import BaseModel 
from src.preprocessing import preprocess_new_data
from src.predict import predict_new_data

'''
class new_property(BaseModel):
   area: int
   property_type: str
   rooms_number: int
   zip_code: int
   land_area: int
   garden: bool
   garden_area: int
   equipped_kitchen: bool
   full_address: str
   swimming_pool: bool
   furnished: bool
   open_fire: bool
   terrace: bool
   terrace_area: int
   facades_number: int
   building_state: str
'''

app = fastapi.FastAPI()

@app.get("/")
def get_request_info():
    return "This route expects a POST request with data of a house in JSON format. The JSON data should have the following fields:\n" \
           "{\n" \
           "  \"data\": {\n" \
           "    \"Living area\": int,\n" \
           "    \"Subtype of property\": \"APARTMENT\" | \"HOUSE\" | \"OTHERS\",\n" \
           "    \"Number of bedrooms\": int,\n" \
           "    \"Zip code\": int,\n" \
           "    \"Garden\": Optional[bool],\n" \
           "    \"Garden area\": Optional[int],\n" \
           "    \"Kitchen\": Optional[bool],\n" \
           "    \"Locality\": Optional[str],\n" \
           "    \"Swimming pool\": Optional[bool],\n" \
           "    \"Terrace\": Optional[bool],\n" \
           "    \"Number of facades\": Optional[int],\n" \
           "    \"State of the building\": Optional[\n" \
           "      \"NEW\" | \"GOOD\" | \"TO RENOVATE\" | \"JUST RENOVATED\" | \"TO REBUILD\"\n" \
           "    ]\n" \
           "  }\n" \
           "}"

@app.post("/predict")
def process_house_data(new_house_data: dict):
    # Call the preprocess_data function from src/preprocessing.py
    preprocessed_data = preprocess_new_data(new_house_data)
    
    # Make predictions for the new house's data
    prediction_result = predict_new_data(preprocessed_data)

    # Return the prediction result
    return prediction_result

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)