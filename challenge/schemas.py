from pydantic import BaseModel, validator
from fastapi import HTTPException, status
from typing import List

VALID_FLIGHT_TYPES = {"I", "N"}

class Flight(BaseModel):
    """
    Represents a single flight's relevant information.
    """
    OPERA: str
    MES: int
    TIPOVUELO: str

    @validator("MES")
    def validate_month(cls, value: int) -> int:
        """
        Ensures the month is between 1 and 12.
        Raises an HTTPException if it's not valid.
        """
        if not (1 <= value <= 12):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MES must be between 1 and 12"
            )
        return value

    @validator("TIPOVUELO")
    def validate_flight_type(cls, value: str) -> str:
        """
        Ensures the flight type is either 'I' or 'N'.
        Raises an HTTPException if it's not valid.
        """
        if value not in VALID_FLIGHT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"TIPOVUELO must be one of {VALID_FLIGHT_TYPES}"
            )
        return value


class PredictionInfo(BaseModel):
    """
    Holds a list of flights for which delay predictions are requested.
    """
    flights: List[Flight]