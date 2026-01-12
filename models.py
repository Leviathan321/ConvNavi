from dataclasses import dataclass, field
import json
from typing import Dict, List
import os
from dotenv import load_dotenv

from car.state import CarState

load_dotenv()


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

    # NEW: persistent POI dialogue state
    poi_constraints: Dict = field(default_factory=dict)

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
