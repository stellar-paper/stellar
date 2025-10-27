from typing import Any, Literal, Optional

from pydantic import BaseModel, RootModel

FuelType = Literal["SP95_E10", "SP95", "DIESEL", "SP98"]


class Coordinates(BaseModel):
    lat: float
    lng: float


LosSorting = Literal[
    "RELEVANCE",
    "DISTANCE",
    "NAME",
    "PRICE",
    # "RATING"  # doesnt seem to be working
]

LosCategory = Literal[
    "airports",
    "atm",
    "bakery",
    "banks",
    "bars_pubs",
    "service",
    "books",
    "cafe",
    "car_wash",
    "charging_stations",
    "chemists",
    "cinemas",
    "convenience_stores",
    "diy_stores",
    "drugstores",
    "fast_food",
    "furniture",
    "garden_center",
    "gymnasium",
    "hair_salon",
    "hospitals",
    "hotels",
    "mini_service",
    "parking",
    "car_park",
    "park_ride",
    "parking_lot",
    "petrol_stations",
    "police",
    "post_offices",
    "public_toilet",
    "rest_area",
    "restaurants",
    "shop_flowers",
    "shopping_center",
    "stadiums",
    "supermarkets",
    "tea_house",
]


class GetLosCoordinatesResponse(BaseModel):
    coordinates: Coordinates
    destination: str
    city: str


class LosFilter(BaseModel):
    name: str


class LosSingleFilter(LosFilter):
    name: str


class LosMultiFilter(LosFilter):
    name: str
    values: list[str]


class LosSearchParams(BaseModel):
    categories: list[LosCategory]
    max_results: int
    max_search_distance: int
    origin_location: list[Coordinates] | Coordinates
    search_location: str | list[Coordinates] | Coordinates
    query: str
    sorting_options: Optional[LosSorting]
    poi_filters: Optional[list[LosFilter]]


class Address(BaseModel):
    countryCode: Optional[str] = None
    country: Optional[str] = None
    regionCode: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    postalCode: Optional[str] = None
    district: Optional[str] = None
    street: Optional[str] = None
    houseNumber: Optional[str] = None


class StructuredHours(BaseModel):
    start: Optional[str] = None
    duration: Optional[str] = None
    recurrence: Optional[str] = None


class BusinessHours(BaseModel):
    formattedHours: Optional[list[str]] = None
    nextStatusChange: Optional[str] = None
    nextStatusChangeTime: Optional[str] = None
    status: Optional[str] = None
    structured: Optional[list[StructuredHours]] = None


class Timeframe(BaseModel):
    start: Optional[str] = None
    duration: Optional[str] = None
    recurrence: Optional[str] = None


class ListPrice(BaseModel):
    service: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    formattedPriceString: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    timeframe: Optional[Timeframe] = None


class PreComputedPrice(BaseModel):
    service: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    formattedPriceString: Optional[str] = None
    unit: Optional[str] = None
    fuelType: Optional[str] = None
    lastUpdate: Optional[str] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    quantity: Optional[float] = None


class PriceSummary(BaseModel):
    priceRangeLevel: Optional[float] = None
    priceRangeText: Optional[str] = None
    priceSummaryText: Optional[str] = None
    free: Optional[bool] = None


class PriceStructured(BaseModel):
    listPrices: Optional[list[ListPrice]] = None
    preComputedPrices: Optional[list[PreComputedPrice]] = None


class Commercial(BaseModel):
    onlinePay: Optional[bool] = None
    supportInvoice: Optional[bool] = None
    preferredPartners: Optional[list[str]] = None
    priceSummary: Optional[PriceSummary] = None
    paymentMethods: Optional[list[str]] = None
    priceStructured: Optional[PriceStructured] = None


class FoodType(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    primary: Optional[bool] = None


class ChargingPoint(BaseModel):
    evseId: Optional[str] = None
    providerChargingPointId: Optional[str] = None
    availability: Optional[str] = None
    plugs: Optional[list[dict]] = None
    position: Optional[Coordinates] = None


class Charging(BaseModel):
    stationOperator: Optional[str] = None
    stationProvider: Optional[str] = None
    authentications: Optional[list[str]] = None
    verificationStatus: Optional[bool] = None
    access: Optional[str] = None
    chargingPoints: Optional[list[ChargingPoint]] = None


class LosPlaceContact(BaseModel):
    email: Optional[str] = None
    phoneNumber: Optional[str] = None
    website: Optional[str] = None


class LosPlaceReviewSummary(BaseModel):
    attributionRequired: Optional[bool] = None
    availableAttributionIcons: Optional[list[dict]] = None
    availableIcons: Optional[list[dict]] = None
    averageRating: Optional[float] = None
    ciVersion: Optional[str] = None
    provider: Optional[str] = None
    ratingIconId: Optional[str] = None
    reviewCount: Optional[int] = None


class LosPlace(BaseModel):
    provider: Optional[str] = None
    providerId: Optional[int] = None
    providerPlaceId: Optional[str] = None
    title: Optional[str] = None
    position: Optional[Coordinates] = None
    address: Optional[Address] = None
    formattedAddress: Optional[str] = None
    entrances: Optional[list[dict]] = None
    placeType: Optional[str] = None
    category: Optional[dict] = None
    secondaryCategories: Optional[list[dict]] = None
    hasChildrenPois: Optional[Literal["YES", "NO"]] = None
    businessHours: Optional[BusinessHours] = None
    reviewSummary: Optional[LosPlaceReviewSummary] = None
    phonemes: Optional[dict] = None
    commercial: Optional[Commercial] = None
    brand: Optional[dict] = None
    foodTypes: Optional[list[FoodType]] = None
    images: Optional[list[dict]] = None
    contact: Optional[LosPlaceContact] = None
    staticIconDiscriminator: Optional[dict] = None
    refueling: Optional[dict] = None
    parking: Optional[dict] = None
    charging: Optional[Charging] = None
    availability: Optional[dict] = None


class LosPlacesSearchResponseItem(BaseModel):
    place: Optional[LosPlace] = None
    distance: Optional[float] = None
    deviateDistance: Optional[int] = None
    deviateTime: Optional[int] = None


class LosDerivedSearchLocation(BaseModel):
    searchLocation: str = None


class LosPlacesSearchResponse(BaseModel):
    translationsVersion: Optional[str] = None
    filtersVersion: Optional[str] = None
    items: list[LosPlacesSearchResponseItem] = None
    derivedSearchLocation: Optional[LosDerivedSearchLocation] = None


class PlaceRanking(BaseModel):
    closest_to_user_location: Optional[str] = None
    best_rated: Optional[str] = None
    cheapest_gas_station: Optional[str] = None


class PriceList(BaseModel):
    service: Optional[str] = None
    price: Optional[str] = None


class RealtimeGasPrice(BaseModel):
    service: Optional[str] = None
    price: Optional[str] = None
    fuelType: Optional[str] = None


class ChargingPointInfo(BaseModel):
    availability: Optional[str] = None
    plugs: Optional[list[dict]] = None
    position: Optional[Coordinates] = None


class BestPlacesRanking(BaseModel):
    closest_to_user_location: str | None = None
    closest_to_location: str | None = None
    best_rated: str | None = None
    cheapest_gas_station_id: str | None = None


class PlacesRanking(RootModel):
    root: dict[str, Any]


class FindPlaceResult(BaseModel):
    places_ranking: PlacesRanking
    places: list[dict[str, Any]]
    los_places: list[LosPlace]
