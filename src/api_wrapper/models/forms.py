from typing import Optional

from pydantic import BaseModel, field_validator

from ..constants.model import VALID_ACTIV_PERM, VALID_SURFACE, VALID_TYPE_FORM


class SingleForm(BaseModel):
    description_activity: str
    other_nature_activity: Optional[str] = None
    precision_act_sec_agricole: Optional[str] = None
    type_form: Optional[str] = None
    nature: Optional[str] = None
    surface: Optional[str] = None
    cj: Optional[str] = None
    activity_permanence_status: Optional[str] = None

    @field_validator("type_form")
    def validate_type_form(cls, v):
        if (v is not None) and (v not in VALID_TYPE_FORM):
            raise ValueError(f"Invalid type_form '{v}', must be one of {VALID_TYPE_FORM}")
        return v

    @field_validator("nature")
    def validate_nature(cls, v):
        if (v is not None) and (not v.isdigit() or len(v) != 2):
            raise ValueError("nature must be a two-digit number (e.g., '01')")
        return v

    @field_validator("surface")
    def validate_surface(cls, v):
        if (v is not None) and (v not in VALID_SURFACE):
            raise ValueError(f"Invalid surface '{v}', must be one of {VALID_SURFACE}")
        return v

    @field_validator("cj")
    def validate_cj(cls, v):
        if (v is not None) and (not v.isdigit() or len(v) != 2):
            raise ValueError("cj must be a 4-digit number (e.g., '5499')")
        return v

    @field_validator("activity_permanence_status")
    def validate_activity_permanence_status(cls, v):
        if (v is not None) and (v not in VALID_ACTIV_PERM):
            raise ValueError(f"Invalid surface '{v}', must be one of {VALID_ACTIV_PERM}")
        return v
