from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    """Base of the stored record models: strict types, no extra fields, defaults validated."""

    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)
