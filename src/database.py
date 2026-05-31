import sqlite3
import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class APODatabase:
    """Manages SQLite storage for APO runs, steps, predictions, and telemetries."""
    
    def __init__(self, db_path: str = "apo_database.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Runs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    train_ratio REAL,
                    val_ratio REAL,
                    test_ratio REAL,
                    seed INTEGER,
                    config_json TEXT
                )
            """)
            
            # 2. Steps Table (each iteration in a run)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    mean_f1 REAL NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id),
                    UNIQUE(run_id, step_index)
                )
            """)
            
            # 3. Predictions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    f1_score REAL NOT NULL,
                    prediction_json TEXT NOT NULL,
                    gold_json TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    candidate_tokens INTEGER,
                    cost REAL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id),
                    UNIQUE(run_id, step_index, file_name)
                )
            """)
            
            # 4. Semantic Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    pred_text TEXT,
                    gold_text TEXT,
                    is_match INTEGER,
                    PRIMARY KEY (pred_text, gold_text)
                )
            """)
            
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def save_run(
        self, 
        run_id: str, 
        timestamp: str, 
        model_name: str, 
        train_ratio: float, 
        val_ratio: float, 
        test_ratio: float, 
        seed: int,
        config_dict: Dict[str, Any]
    ):
        """Saves run details to the runs table. Overwrites if exists (e.g. on resume)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO runs 
                (run_id, timestamp, model_name, train_ratio, val_ratio, test_ratio, seed, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, 
                    timestamp, 
                    model_name, 
                    train_ratio, 
                    val_ratio, 
                    test_ratio, 
                    seed, 
                    json.dumps(config_dict)
                )
            )
            conn.commit()

    def save_step(self, run_id: str, step_index: int, prompt: str, mean_f1: float, status: str):
        """Saves a single optimization trajectory step."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO steps (run_id, step_index, prompt, mean_f1, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, step_index, prompt, mean_f1, status)
            )
            conn.commit()

    def save_prediction(
        self, 
        run_id: str, 
        step_index: int, 
        file_name: str, 
        f1_score: float, 
        prediction_dict: Dict[str, Any], 
        gold_dict: Dict[str, Any],
        prompt_tokens: int,
        candidate_tokens: int,
        cost: float
    ):
        """Saves or updates a structured prediction with token costs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions 
                (run_id, step_index, file_name, f1_score, prediction_json, gold_json, prompt_tokens, candidate_tokens, cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, 
                    step_index, 
                    file_name, 
                    f1_score, 
                    json.dumps(prediction_dict), 
                    json.dumps(gold_dict),
                    prompt_tokens,
                    candidate_tokens,
                    cost
                )
            )
            conn.commit()

    def get_prediction(self, run_id: str, step_index: int, file_name: str) -> Optional[Tuple[Dict[str, Any], float, int, int, float]]:
        """Retrieves cached prediction info if it exists (for resumability)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT prediction_json, f1_score, prompt_tokens, candidate_tokens, cost 
                FROM predictions 
                WHERE run_id = ? AND step_index = ? AND file_name = ?
                """,
                (run_id, step_index, file_name)
            )
            row = cursor.fetchone()
            if row:
                return (
                    json.loads(row["prediction_json"]),
                    row["f1_score"],
                    row["prompt_tokens"],
                    row["candidate_tokens"],
                    row["cost"]
                )
        return None

    def get_step(self, run_id: str, step_index: int) -> Optional[Dict[str, Any]]:
        """Retrieves a cached step instruction and score."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT prompt, mean_f1, status FROM steps WHERE run_id = ? AND step_index = ?",
                (run_id, step_index)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_run_steps(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieves the full step trajectory for plotting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT step_index, mean_f1, status, prompt FROM steps WHERE run_id = ? ORDER BY step_index ASC",
                (run_id,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
            
    def get_run_predictions(self, run_id: str, step_index: int) -> List[Dict[str, Any]]:
        """Retrieves all predictions for a step (to feed to mutator on resume)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT file_name, f1_score, prediction_json, gold_json 
                FROM predictions 
                WHERE run_id = ? AND step_index = ?
                """,
                (run_id, step_index)
            )
            rows = cursor.fetchall()
            return [
                {
                    "name": r["file_name"],
                    "score": r["f1_score"],
                    "prediction": json.loads(r["prediction_json"]),
                    "gold": json.loads(r["gold_json"])
                }
                for r in rows
            ]

    def get_semantic_match(self, pred_text: str, gold_text: str) -> Optional[bool]:
        """Retrieves cached semantic evaluation result."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT is_match FROM semantic_cache WHERE pred_text = ? AND gold_text = ?",
                (pred_text, gold_text)
            )
            row = cursor.fetchone()
            if row is not None:
                return bool(row["is_match"])
        return None

    def save_semantic_match(self, pred_text: str, gold_text: str, is_match: bool):
        """Saves a semantic evaluation result to the cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO semantic_cache (pred_text, gold_text, is_match) VALUES (?, ?, ?)",
                (pred_text, gold_text, int(is_match))
            )
            conn.commit()
