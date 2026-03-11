"""Face recognition module for encoding and matching faces.

Uses the face_recognition library (built on dlib) to extract face embeddings
and compare them for identification.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import face_recognition
except ImportError as e:
    raise ImportError(
        "To use face recognition, please install the extra dependencies: "
        "pip install '.[face_recognition]'"
    ) from e


logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognition using the face_recognition library."""

    def __init__(self, model: str = "hog", upsample: int = 1) -> None:
        """Initialize the face recognizer.

        Args:
            model: Face detection model to use. Options:
                   - "hog": Faster, less accurate (good for CPU)
                   - "cnn": Slower, more accurate (requires GPU/dlib with CUDA)
            upsample: How many times to upsample image when detecting faces (higher = find smaller faces)

        """
        self.model = model
        self.upsample = upsample
        logger.info(f"Face recognizer initialized with model={model}, upsample={upsample}")

    def get_face_locations(
        self,
        frame: NDArray[np.uint8],
    ) -> List[Tuple[int, int, int, int]]:
        """Detect face locations in a frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of face bounding boxes as (top, right, bottom, left) tuples.

        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = frame[:, :, ::-1]

        try:
            locations = face_recognition.face_locations(
                rgb_frame,
                model=self.model,
                number_of_times_to_upsample=self.upsample,
            )
            logger.debug(f"face_recognition detected {len(locations)} face(s)")
            return locations
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def get_face_encodings(
        self,
        frame: NDArray[np.uint8],
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[NDArray[np.float64]]:
        """Extract 128-dimensional face encodings from a frame.

        Args:
            frame: BGR image from OpenCV.
            face_locations: Optional pre-computed face locations. If None,
                           faces will be detected automatically.

        Returns:
            List of 128-dimensional numpy arrays (one per face detected).

        """
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1]

        try:
            if face_locations is None:
                face_locations = face_recognition.face_locations(
                    rgb_frame,
                    model=self.model,
                    number_of_times_to_upsample=self.upsample,
                )

            if not face_locations:
                return []

            encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations,
            )
            logger.debug(f"Extracted {len(encodings)} face encoding(s)")
            return encodings

        except Exception as e:
            logger.error(f"Error extracting face encodings: {e}")
            return []

    def get_single_face_encoding(
        self,
        frame: NDArray[np.uint8],
    ) -> Optional[NDArray[np.float64]]:
        """Get encoding for the largest/most prominent face in frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            128-dimensional numpy array for the largest face, or None if no face found.

        """
        rgb_frame = frame[:, :, ::-1].copy()  # Ensure contiguous array
        h, w = frame.shape[:2]
        logger.debug(f"Processing frame of size {w}x{h}")

        try:
            # Let face_recognition handle detection and encoding together
            # This avoids passing face_locations which causes compatibility issues
            encodings = face_recognition.face_encodings(rgb_frame)

            logger.debug(f"face_recognition.face_encodings found {len(encodings)} face(s)")

            if not encodings:
                logger.debug("No faces detected by face_recognition library")
                return None

            # If multiple faces, we just return the first one (usually largest/most prominent)
            if len(encodings) > 1:
                logger.debug(f"Multiple faces found, using first of {len(encodings)}")

            logger.debug("Successfully extracted face encoding")
            return encodings[0]

        except Exception as e:
            logger.error(f"Error getting single face encoding: {e}")
            return None

    def compare_faces(
        self,
        known_encoding: NDArray[np.float64],
        unknown_encoding: NDArray[np.float64],
        tolerance: float = 0.6,
    ) -> Tuple[bool, float]:
        """Compare two face encodings.

        Args:
            known_encoding: The known face encoding to compare against.
            unknown_encoding: The unknown face encoding to check.
            tolerance: Maximum distance for a match. Lower = stricter.

        Returns:
            Tuple of (is_match, distance).

        """
        distance = float(np.linalg.norm(known_encoding - unknown_encoding))
        is_match = distance <= tolerance
        return is_match, distance

    def identify_faces_in_frame(
        self,
        frame: NDArray[np.uint8],
        known_encodings: List[NDArray[np.float64]],
        known_names: List[str],
        tolerance: float = 0.6,
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Identify all faces in a frame against known faces.

        Args:
            frame: BGR image from OpenCV.
            known_encodings: List of known face encodings.
            known_names: List of names corresponding to known_encodings.
            tolerance: Maximum distance for a match.

        Returns:
            List of (name, distance, location) tuples for each face found.
            Name is "Unknown" if no match found.

        """
        rgb_frame = frame[:, :, ::-1]

        try:
            face_locations = face_recognition.face_locations(rgb_frame, model=self.model)

            if not face_locations:
                return []

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []

            for encoding, location in zip(face_encodings, face_locations):
                best_match = "Unknown"
                best_distance = float("inf")

                for known_enc, name in zip(known_encodings, known_names):
                    distance = float(np.linalg.norm(known_enc - encoding))
                    if distance < best_distance and distance <= tolerance:
                        best_distance = distance
                        best_match = name

                results.append((best_match, best_distance, location))

            return results

        except Exception as e:
            logger.error(f"Error identifying faces: {e}")
            return []
