"""API request/response schemas."""

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    """Request payload for the research question endpoint."""

    question: str = Field(min_length=3, max_length=10000)
    document_ids: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_hyde: bool = False
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)


    @model_validator(mode="after")
    def validate_year_range(self):
        """Reject an inverted year range at request-validation time."""
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be less than or equal to year_to")
        return self
