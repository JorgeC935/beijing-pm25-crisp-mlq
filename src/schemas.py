from pydantic import BaseModel, ConfigDict, Field, field_validator

NUMERIC = ('pm25','pm10','so2','no2','co','o3','temp','pres','dewp','rain','wspm')
MEASUREMENTS = (*NUMERIC, 'wd')

class Observation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    pm25: float = Field(ge=0)
    pm10: float | None = Field(None, ge=0)
    so2: float | None = Field(None, ge=0)
    no2: float | None = Field(None, ge=0)
    co: float | None = Field(None, ge=0)
    o3: float | None = Field(None, ge=0)
    temp: float | None = None
    pres: float | None = Field(None, gt=0)
    dewp: float | None = None
    rain: float | None = Field(None, ge=0)
    wspm: float | None = Field(None, ge=0)
    wd: str = 'Desconocido'

    @field_validator('wd')
    @classmethod
    def normalize_wind(cls, v):
        return v.strip() or 'Desconocido'
