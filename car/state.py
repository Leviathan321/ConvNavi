from enum import Enum
from typing import Dict, Any


class WindowState(Enum):
    OPEN = "open"
    CLOSED = "closed"


class HeadlightState(Enum):
    OFF = "off"
    LOW = "low"
    HIGH = "high"


class LightState(Enum):
    OFF = "off"
    ON = "on"


class AmbientLightLevel(Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DoorState(Enum):
    OPEN = "open"
    CLOSED = "closed"


class WiperState(Enum):
    OFF = "off"
    INTERMITTENT = "intermittent"
    LOW = "low"
    HIGH = "high"


class ClimateMode(Enum):
    AUTO = "auto"
    MANUAL = "manual"


class SeatHeatingLevel(Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CarState:
    """
    Holds the current state of vehicle functions, including windows,
    lights, doors, climate, wipers, and seat heating.
    """

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "windows": {
                "front_left": WindowState.CLOSED,
                "front_right": WindowState.CLOSED,
                "rear_left": WindowState.CLOSED,
                "rear_right": WindowState.CLOSED,
            },
            "lights": {
                "headlights": HeadlightState.OFF,
                "fog_lights": LightState.OFF,
                "interior_front": LightState.OFF,
                "interior_rear": LightState.OFF,
                "ambient": AmbientLightLevel.OFF,
            },
            "doors": {
                "front_left": DoorState.CLOSED,
                "front_right": DoorState.CLOSED,
                "rear_left": DoorState.CLOSED,
                "rear_right": DoorState.CLOSED,
                "trunk": DoorState.CLOSED,
            },
            "climate": {
                "temperature_c": 21.0,
                "fan_level": 2,
                "mode": ClimateMode.AUTO,
            },
            "wipers": {
                "state": WiperState.OFF,
            },
            "seat_heating": {
                "driver": SeatHeatingLevel.OFF,
                "front_passenger": SeatHeatingLevel.OFF,
            }
        }

    def get_state(self):
        def normalize(v):
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, dict):
                return {k: normalize(val) for k, val in v.items()}
            return v

        return normalize(self.state)
    
    def get(self, domain: str, key: str) -> Any:
        return self.state[domain][key]

    def set(self, domain: str, key: str, value: Any) -> None:
        self.state[domain][key] = value

ENUM_MAP = {
    "windows": {
        "front_left": WindowState,
        "front_right": WindowState,
        "rear_left": WindowState,
        "rear_right": WindowState,
    },
    "lights": {
        "headlights": HeadlightState,
        "fog_lights": LightState,
        "interior_front": LightState,
        "interior_rear": LightState,
        "ambient": AmbientLightLevel,
    },
    "doors": {
        "front_left": DoorState,
        "front_right": DoorState,
        "rear_left": DoorState,
        "rear_right": DoorState,
        "trunk": DoorState,
    },
    "climate": {
        "mode": ClimateMode,
        "temperature_c": float,
        "fan_level": int,
    },
    "wipers": {
        "state": WiperState,
    },
    "seat_heating": {
        "driver": SeatHeatingLevel,
        "front_passenger": SeatHeatingLevel,
    },
}

POSSIBLE_CAR_VALUES = {
    "windows": [e.value for e in WindowState],
    "headlights": [e.value for e in HeadlightState],
    "fog_lights": [e.value for e in LightState],
    "interior_front": [e.value for e in LightState],
    "interior_rear": [e.value for e in LightState],
    "ambient": [e.value for e in AmbientLightLevel],
    "doors": [e.value for e in DoorState],
    "wipers": [e.value for e in WiperState],
    "climate_mode": [e.value for e in ClimateMode],
    "seat_heating": [e.value for e in SeatHeatingLevel],
    "temperature_c": list(range(16, 29)),
    "fan_level": list(range(0, 6)),
}