from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, status

from challenge.model import DelayModel
from challenge.schemas import PredictionInfo

app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0.0",
    description="API to predict the probability of delay for a flight taking off or landing at SCL airport.",
)

delay_model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health-check endpoint to verify the service status.
    Returns a JSON object with a status 'OK'.
    """
    return {"status": "OK"}


@app.post("/predict", status_code=status.HTTP_200_OK, response_model=dict)
async def post_predict(input_data: PredictionInfo) -> dict:
    """
    Generates delay predictions for the provided set of flights.
    :param input_data: A PredictionInfo object containing a list of flights.
    :return: A dictionary with the predictions.
    """
    try:
        data = [flight.dict() for flight in input_data.flights]
        df = pd.DataFrame(data)

        preprocessed_data = delay_model.preprocess(df)
        predictions = delay_model.predict(preprocessed_data)

        return {"predict": predictions}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the prediction: {str(e)}",
        )
