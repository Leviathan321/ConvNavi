from enum import Enum
from typing import Dict, Any


# ----------------- ENUM DEFINITIONS -----------------

class WindowState(Enum):
    OPEN = "open"
    CLOSED = "closed"

class LightState(Enum):
    OFF = "off"
    ON = "on"

class ClimateState(Enum):
    OFF = "off"
    ON = "on"

class DoorState(Enum):
    OPEN = "open"
    CLOSED = "closed"

class WiperState(Enum):
    OFF = "off"
    INTERMITTENT = "intermittent"
    LOW = "low"
    HIGH = "high"

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
    Holds the current state of vehicle functions.

    The ``windows``, ``lights``, ``climate`` and ``seat_heating`` subsystems mirror
    the LUNAR interaction ``CarState`` (LUNAR-DEV/llm/model/interaction.py): same
    target names and same allowed values. ``doors``, ``wipers`` and ``media`` are
    ConvNavi-specific subsystems with no LUNAR equivalent.
    """

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            # --- Subsystems aligned with the LUNAR interaction CarState ---
            "windows": {
                "window_front_left": WindowState.CLOSED,
                "window_front_right": WindowState.CLOSED,
                "window_rear_left": WindowState.CLOSED,
                "window_rear_right": WindowState.CLOSED,
            },
            "lights": {
                "fog_light": LightState.OFF,
                "head_light": LightState.OFF,
                "ambient_light": LightState.OFF,
                "reading_light_front_left": LightState.OFF,
                "reading_light_front_right": LightState.OFF,
                "reading_light_rear_left": LightState.OFF,
                "reading_light_rear_right": LightState.OFF,
            },
            "climate": {
                "temperature": 21.0,
                "climate": ClimateState.OFF,
                "fan": ClimateState.OFF,
            },
            "seat_heating": {
                "seat_heating_front_left": SeatHeatingLevel.OFF,
                "seat_heating_front_right": SeatHeatingLevel.OFF,
                "seat_heating_rear_left": SeatHeatingLevel.OFF,
                "seat_heating_rear_right": SeatHeatingLevel.OFF,
            },
            # --- ConvNavi-specific subsystems (no LUNAR equivalent) ---
            "doors": {
                "front_left": DoorState.CLOSED,
                "front_right": DoorState.CLOSED,
                "rear_left": DoorState.CLOSED,
                "rear_right": DoorState.CLOSED,
                "trunk": DoorState.CLOSED,
            },
            "wipers": {
                "state": WiperState.OFF,
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
        "window_front_left": WindowState,
        "window_front_right": WindowState,
        "window_rear_left": WindowState,
        "window_rear_right": WindowState,
    },
    "lights": {
        "fog_light": LightState,
        "head_light": LightState,
        "ambient_light": LightState,
        "reading_light_front_left": LightState,
        "reading_light_front_right": LightState,
        "reading_light_rear_left": LightState,
        "reading_light_rear_right": LightState,
    },
    "climate": {
        "temperature": float,
        "climate": ClimateState,
        "fan": ClimateState,
    },
    "seat_heating": {
        "seat_heating_front_left": SeatHeatingLevel,
        "seat_heating_front_right": SeatHeatingLevel,
        "seat_heating_rear_left": SeatHeatingLevel,
        "seat_heating_rear_right": SeatHeatingLevel,
    },
    "doors": {
        "front_left": DoorState,
        "front_right": DoorState,
        "rear_left": DoorState,
        "rear_right": DoorState,
        "trunk": DoorState,
    },
    "wipers": {
        "state": WiperState,
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
    "fog_light": [e.value for e in LightState],
    "head_light": [e.value for e in LightState],
    "ambient_light": [e.value for e in LightState],
    "reading_light": [e.value for e in LightState],
    "temperature": list(range(16, 29)),
    "climate": [e.value for e in ClimateState],
    "fan": [e.value for e in ClimateState],
    "seat_heating": [e.value for e in SeatHeatingLevel],
    "doors": [e.value for e in DoorState],
    "wipers": [e.value for e in WiperState],
    "media_state": [e.value for e in RadioState],
    "volume": list(range(0, 11)),
    "radio_station": [round(x * 0.1, 1) for x in range(880, 1081)],
    "music_genre": [e.value for e in MusicGenre],  # 88.0 - 108.0 FM
}
