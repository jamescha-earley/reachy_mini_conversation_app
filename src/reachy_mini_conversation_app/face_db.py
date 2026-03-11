"""SQLite database manager for facial recognition embeddings.

Provides persistent storage for face embeddings and associated metadata,
enabling the robot to recognize and remember people across sessions.
"""

import sqlite3
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class FaceDatabase:
    """Thread-safe SQLite database for storing face embeddings."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the face database.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.

        """
        if db_path is None:
            # Default to user's home directory
            db_dir = Path.home() / ".reachy_mini"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "faces.db")

        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"Face database initialized at {self.db_path}")

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        embedding BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)
                """)
                conn.commit()
            finally:
                conn.close()

    def _embedding_to_bytes(self, embedding: NDArray[np.float64]) -> bytes:
        """Convert numpy embedding array to bytes for storage."""
        return embedding.tobytes()

    def _bytes_to_embedding(self, data: bytes) -> NDArray[np.float64]:
        """Convert stored bytes back to numpy embedding array."""
        return np.frombuffer(data, dtype=np.float64)

    def add_face(self, name: str, embedding: NDArray[np.float64]) -> Dict[str, Any]:
        """Add or update a face in the database.

        Args:
            name: Name/identifier for the person.
            embedding: 128-dimensional face encoding from face_recognition library.

        Returns:
            Dict with status and face_id.

        """
        if embedding.shape != (128,):
            return {"error": f"Invalid embedding shape: {embedding.shape}, expected (128,)"}

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                embedding_bytes = self._embedding_to_bytes(embedding)
                now = datetime.now().isoformat()

                # Try to update existing face first
                cursor.execute(
                    "UPDATE faces SET embedding = ?, updated_at = ? WHERE name = ?",
                    (embedding_bytes, now, name),
                )

                if cursor.rowcount == 0:
                    # Insert new face
                    cursor.execute(
                        "INSERT INTO faces (name, embedding, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (name, embedding_bytes, now, now),
                    )
                    face_id = cursor.lastrowid
                    action = "added"
                else:
                    cursor.execute("SELECT id FROM faces WHERE name = ?", (name,))
                    face_id = cursor.fetchone()[0]
                    action = "updated"

                conn.commit()
                logger.info(f"Face {action} for '{name}' (id={face_id})")
                return {"status": "success", "action": action, "face_id": face_id, "name": name}

            except sqlite3.Error as e:
                logger.error(f"Database error adding face: {e}")
                return {"error": str(e)}
            finally:
                conn.close()

    def get_face(self, name: str) -> Optional[Tuple[int, str, NDArray[np.float64]]]:
        """Get a face by name.

        Args:
            name: Name of the person to look up.

        Returns:
            Tuple of (id, name, embedding) or None if not found.

        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, name, embedding FROM faces WHERE name = ?",
                    (name,),
                )
                row = cursor.fetchone()
                if row:
                    return (row[0], row[1], self._bytes_to_embedding(row[2]))
                return None
            finally:
                conn.close()

    def get_all_faces(self) -> List[Tuple[int, str, NDArray[np.float64]]]:
        """Get all faces from the database.

        Returns:
            List of (id, name, embedding) tuples.

        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, embedding FROM faces ORDER BY name")
                rows = cursor.fetchall()
                return [(row[0], row[1], self._bytes_to_embedding(row[2])) for row in rows]
            finally:
                conn.close()

    def list_faces(self) -> List[Dict[str, Any]]:
        """List all registered faces (without embeddings).

        Returns:
            List of dicts with id, name, created_at, updated_at.

        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, name, created_at, updated_at FROM faces ORDER BY name"
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "created_at": row[2],
                        "updated_at": row[3],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def delete_face(self, name: str) -> Dict[str, Any]:
        """Delete a face from the database.

        Args:
            name: Name of the person to delete.

        Returns:
            Dict with status.

        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Deleted face for '{name}'")
                    return {"status": "success", "deleted": name}
                else:
                    return {"error": f"Face '{name}' not found"}

            except sqlite3.Error as e:
                logger.error(f"Database error deleting face: {e}")
                return {"error": str(e)}
            finally:
                conn.close()

    def find_matching_face(
        self,
        embedding: NDArray[np.float64],
        tolerance: float = 0.6,
    ) -> Optional[Tuple[str, float]]:
        """Find a matching face in the database.

        Args:
            embedding: 128-dimensional face encoding to match.
            tolerance: Maximum distance for a match (lower = stricter). Default 0.6.

        Returns:
            Tuple of (name, distance) for best match, or None if no match found.

        """
        all_faces = self.get_all_faces()
        if not all_faces:
            return None

        best_match: Optional[Tuple[str, float]] = None
        best_distance = float("inf")

        for _, name, stored_embedding in all_faces:
            # Calculate Euclidean distance between embeddings
            distance = float(np.linalg.norm(embedding - stored_embedding))

            if distance < best_distance and distance <= tolerance:
                best_distance = distance
                best_match = (name, distance)

        return best_match

    def count_faces(self) -> int:
        """Get the number of registered faces."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM faces")
                return cursor.fetchone()[0]
            finally:
                conn.close()
