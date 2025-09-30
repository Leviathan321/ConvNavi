from dataclasses import asdict, dataclass, field
import json
from typing import Dict, List
import os
from dotenv import load_dotenv

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
    tokens: dict = field(default_factory=list)

    def add_turn(self, turn: Turn):
        if len(self.turns) >= self.max_turns:
            raise Exception("Max number of turns already reached.")
        self.turns.append(turn)

    def get_history(self, indent: int = 2) -> str:
        return json.dumps([asdict(turn) for turn in self.turns], indent=indent)

    def get_last_turn(self):
        return self.turns[-1]
    
    def get_last_question(self):
        return self.turns[-1].question
    
    def complete(self, response, retrieved_pois):
        if self.turns[-1].question is not None and self.turns[-1].answer is None: 
            self.turns[-1].answer = response
            self.turns[-1].retrieved_pois = retrieved_pois
        else:
            print("Question already answered or no question available.")
            raise Exception()
    def len(self):
        return len(self.turns)
    
    def is_empty(self):
        return len(self.turns) == 0

class SessionManager:
    _instance = None

    def __init__(self):
        if SessionManager._instance is not None:
            raise Exception("Use SessionManager.get_instance() instead of direct instantiation.")
        self.sessions: Dict[int, Session] = {}
        self.current_id = 0
        self.active_session: Session = self.create_session()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        """Get the singleton instance of SessionManager."""
        if cls._instance is None:
            cls._instance = SessionManager()
        return cls._instance

    def create_session(self) -> Session:
        self.current_id += 1
        session = Session(id=self.current_id)
        self.sessions[self.current_id] = session
        self.active_session = session
        return session

    def get_active_session(self) -> Session:
        return self.active_session