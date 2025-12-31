from fastapi import FastAPI 
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
import pandas as pd
from model import recommend, output_recommended_recipes
dataset = pd.read_csv("data/dataset.csv", compression="gzip")

app = FastAPI()

class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False


class PredictionIn(BaseModel):
    nutrition_input:str = Field(..., min_length=1, max_length=100)
    ingredients: List[str] = []
    params: Optional[Params] = None


class Recipe(BaseModel):
    Name:str
    CookTime:str
    PrepTime:str
    TotalTime:str
    RecipeIngredientParts:list[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict/", response_model=PredictionOut)
def update_item(prediction_input: PredictionIn):

    params = (
        prediction_input.params.model_dump()
        if prediction_input.params is not None
        else {}
    )

    recommendation_dataframe = recommend(
        dataset,
        prediction_input.nutrition_input,
        prediction_input.ingredients,
        params
    )

    output = output_recommended_recipes(recommendation_dataframe)

    return {"output": output}
