from enum import Enum
from typing import Dict, Any


# ----------------- ENUM DEFINITIONS -----------------

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

class RadioState(Enum):
    OFF = "off"
    ON = "on"

class MusicGenre(Enum):
    POP = "pop"
    ROCK = "rock"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    HIPHOP = "hiphop"
    ELECTRONIC = "electronic"

# ----------------- CAR STATE -----------------
class CarState:
    """
    Holds the current state of vehicle functions, including windows,
    lights, doors, climate, wipers, seat heating, and media.
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
            },
            "media": {
                "volume": 5,
                "radio_state": RadioState.OFF,
                "radio_station": 101.1,
                "music_genre": MusicGenre.POP,  # Always valid
            }
        }

    # ----------------- NORMALIZATION -----------------
    def get_state(self):
        """Return all states with enums converted to values recursively."""
        def normalize(v):
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, dict):
                return {k: normalize(val) for k, val in v.items()}
            return v

        return normalize(self.state)

    # ----------------- GENERIC GET/SET -----------------
    def get(self, domain: str, key: str) -> Any:
        return self.state[domain][key]

    def set(self, domain: str, key: str, value: Any) -> None:
        self.state[domain][key] = value
# ----------------- ENUM MAP -----------------

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
    "media": {
        "radio_state": RadioState,
        "volume": int,
        "radio_station": float,
        "music_genre": MusicGenre,
    }
}

# ----------------- POSSIBLE VALUES -----------------

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
    "media_state": [e.value for e in RadioState],
    "volume": list(range(0, 11)),
    "radio_station": [round(x * 0.1, 1) for x in range(880, 1081)],
    "music_genre": [e.value for e in MusicGenre],  # 88.0 - 108.0 FM
}
