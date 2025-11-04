from pydantic import BaseModel, Field

class PropertyInput(BaseModel):
    neighbourhood_group: str = Field(..., examples=["Manhattan"])
    neighbourhood: str | None = Field(None, examples=["Harlem"])
    room_type: str = Field(..., examples=["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    latitude: float = Field(..., ge=40.4, le=41.0)
    longitude: float = Field(..., ge=-74.3, le=-73.6)
    minimum_nights: int = Field(..., ge=1, le=60)
    number_of_reviews: int = Field(..., ge=0, le=1000)
    reviews_per_month: float = Field(..., ge=0.0, le=30.0)
    calculated_host_listings_count: int = Field(..., ge=0, le=1000)
    availability_365: int = Field(..., ge=0, le=365)

class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
