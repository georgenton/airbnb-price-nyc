from pydantic import BaseModel, Field

class PropertyInput(BaseModel):
    neighbourhood_group: str = Field(..., examples=["Manhattan"])
    neighbourhood: str | None = Field(None, examples=["Harlem"])
    room_type: str = Field(..., examples=["Entire home/apt","Private room","Shared room","Hotel room"])
    latitude: float = Field(..., ge=40.4, le=41.0, examples=[40.81])
    longitude: float = Field(..., ge=-74.3, le=-73.6, examples=[-73.95])
    minimum_nights: int = Field(..., ge=1, le=60, examples=[3])
    number_of_reviews: int = Field(..., ge=0, le=1000, examples=[25])
    reviews_per_month: float = Field(..., ge=0.0, le=30.0, examples=[1.2])
    calculated_host_listings_count: int = Field(..., ge=0, le=1000, examples=[1])
    availability_365: int = Field(..., ge=0, le=365, examples=[120])

class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
