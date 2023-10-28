import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, FieldValidationInfo

from .validators import (
    validate_log_level,
    str2path,
    validate_enabled,
    validate_not_enabled,
)
from .Enums import ModelProcessor, ModelType


class Testing(BaseModel):
    enabled: bool = Field(False)
    substitutions: Dict[str, str] = Field(default_factory=dict)


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)

    _v = field_validator("enabled", mode="before")(validate_enabled)


class DefaultNotEnabled(BaseModel):
    enabled: bool = Field(False)

    _v = field_validator("enabled", mode="before")(validate_not_enabled)


class LoggingLevelBase(BaseModel):
    level: Optional[int] = None

    _validate_log_level = field_validator("level", mode="before")(
        validate_log_level
    )


class LoggingSettings(LoggingLevelBase):
    class ConsoleLogging(DefaultEnabled, LoggingLevelBase):
        pass

    class SyslogLogging(DefaultNotEnabled, LoggingLevelBase):
        address: Optional[str] = Field("")

    class FileLogging(DefaultEnabled, LoggingLevelBase):
        path: Path = Field("/var/log/zm")
        filename_prefix: str = Field("zmml")
        file_name: Optional[str] = None
        user: str = Field(default="www-data")
        group: str = Field(default="www-data")

        _validate_path = field_validator("path", mode="before")(str2path)

    class SanitizeLogging(DefaultNotEnabled):
        replacement_str: str = Field(default="<sanitized>")

    class IntegrateZMLogging(DefaultNotEnabled):
        debug_level: int = Field(default=4)

    level: int = logging.INFO
    console: ConsoleLogging = Field(default_factory=ConsoleLogging)
    syslog: SyslogLogging = Field(default_factory=SyslogLogging)
    integrate_zm: IntegrateZMLogging = Field(default_factory=IntegrateZMLogging)
    file: FileLogging = Field(default_factory=FileLogging)
    sanitize: SanitizeLogging = Field(default_factory=SanitizeLogging)


class Result(BaseModel):
    label: str
    confidence: float
    bounding_box: List[int]

    def __eq__(self, other):
        if not isinstance(other, Result):
            return False
        return (
            self.label == other.label
            and self.confidence == other.confidence
            and self.bounding_box == other.bounding_box
        )

    def __str__(self):
        return f"'{self.label}' ({self.confidence:.2f}) @ {self.bounding_box}"

    def __repr__(self):
        return f"<'{self.label}' ({self.confidence * 100:.2f}%) @ {self.bounding_box}>"


class DetectionResults(BaseModel, arbitrary_types_allowed=True):
    success: bool = Field(...)
    name: str = Field(...)
    type: ModelType = Field(...)
    processor: ModelProcessor = Field(...)
    results: Optional[List[Result]] = Field(None)
    removed_by_filters: Optional[List[Result]] = Field(None)

    image: Optional[np.ndarray] = Field(None, repr=False)
    extra_image_data: Optional[Dict[str, Any]] = Field(None, repr=False)

    def get_labels(self) -> List[Optional[str]]:
        if not self.results or self.results is None:
            return []
        return (
            [r.label for r in self.results],
            [r.confidence for r in self.results],
            [r.bounding_box for r in self.results],
        )
