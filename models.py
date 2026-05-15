from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel

from car.state import CarState

load_dotenv()


# ==================== CONTEXT MODELS ====================
class POIPreferences(BaseModel):
    """Default POI preferences for the user."""
    category: Optional[str] = None
    cuisine: Optional[str] = None
    price_level: Optional[str] = None
    radius_km: Optional[float] = None
    open_now: Optional[bool] = None
    rating: Optional[float] = None
    parking: Optional[bool] = None
    name: Optional[str] = None


class PersonInfo(BaseModel):
    """Personal information about the user."""
    age: Optional[str] = None
    name: Optional[str] = None
    home: Optional[str] = None
    nationality: Optional[str] = None


class Preferences(BaseModel):
    """User preferences for POI."""
    poi: POIPreferences = POIPreferences()


class Context(BaseModel):
    """Complete user context for navigation."""
    preferences: Preferences = Preferences()
    location: Optional[str] = None
    person: PersonInfo = PersonInfo()

    def to_dict(self) -> dict:
        """Convert context to dictionary."""
        return self.model_dump(exclude_none=False)

    def to_json(self) -> str:
        """Convert context to JSON string."""
        return self.model_dump_json()


@dataclass
class Turn(object):
    question: str
    answer: str
    retrieved_pois: List[dict]

@dataclass
class Session(object):
    id: int
    turns: list = field(default_factory=list)
    max_turns: int = int(os.getenv("MAX_TURNS"))
    tokens: dict = field(default_factory=dict)
    car_state: CarState = field(default_factory=CarState)
    ended_by_user: bool = False
    episodic_memory_stored: bool = False

    # NEW: persistent POI dialogue state
    poi_constraints: Dict = field(default_factory=dict)

    # NEW: user context with preferences and person info
    user_context: Context = field(default_factory=Context)

    def add_turn(self, turn: Turn):
        if len(self.turns) >= self.max_turns:
            raise Exception("Max number of turns already reached.")
        self.turns.append(turn)

    def get_history(self, indent: int = 2) -> str:
        return json.dumps(
            [{"question": t.question, "answer": t.answer} for t in self.turns],
            indent=indent,
        )

    def complete(self, response, retrieved_pois):
        if self.turns and self.turns[-1].answer is None:
            self.turns[-1].answer = response
            self.turns[-1].retrieved_pois = retrieved_pois
        else:
            raise Exception("No open turn to complete.")

    def get_last_retrieved_pois(self) -> List[dict]:
        """Return POIs from the most recent completed turn that had results."""
        for turn in reversed(self.turns):
            if turn.retrieved_pois:
                print("Last retrieved POIs:", turn.retrieved_pois)
                return turn.retrieved_pois
        return []

    def len(self):
        return len(self.turns)

    def is_empty(self):
        return len(self.turns) == 0


class SessionManager:
    _instance = None

    def __init__(self):
        if SessionManager._instance is not None:
            raise Exception("Use SessionManager.get_instance()")
        self.sessions: Dict[str, Session] = {}
        self.current_id = 0

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = SessionManager()
        return cls._instance

    def get_session(self, user_id: str) -> Session:
        return self.sessions.get(user_id, None)

    def create_session(self, user_id: str) -> Session:
        self.current_id += 1
        session = Session(id=self.current_id)
        self.sessions[user_id] = session
        return session