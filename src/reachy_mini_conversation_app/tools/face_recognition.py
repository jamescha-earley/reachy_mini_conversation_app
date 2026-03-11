"""Face recognition tools for LLM function calling.

Provides tools for the robot to register, identify, and manage known faces
through natural conversation.
"""

import os
import time
import logging
from typing import Any, Dict, List

import numpy as np

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Try to import image loading libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Default folder for face photos (relative to project root)
FACES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "faces")


class RegisterFace(Tool):
    """Register a new face with a name for future recognition."""

    name = "register_face"
    description = (
        "Register the face of the person currently in view with their name. "
        "Use this when someone introduces themselves or asks to be remembered. "
        "The person should be looking at the camera when registering."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the person to register",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Register a face from the current camera frame.

        Takes multiple samples over ~1 second and averages them for better accuracy.
        """
        name = (kwargs.get("name") or "").strip()
        if not name:
            return {"error": "Name must be provided"}

        logger.info(f"Tool call: register_face name={name}")

        # Check dependencies
        if deps.camera_worker is None:
            logger.error("register_face: Camera worker not available")
            return {"error": "Camera not available"}

        if deps.face_db is None:
            logger.error("register_face: Face database not initialized")
            return {"error": "Face database not initialized"}

        if deps.face_recognizer is None:
            logger.error("register_face: Face recognizer not available")
            return {"error": "Face recognition not available"}

        # Collect multiple encodings for better accuracy
        encodings: List[np.ndarray] = []
        num_samples = 5
        sample_delay = 0.2  # 200ms between samples

        logger.debug(f"register_face: Collecting {num_samples} samples...")

        for i in range(num_samples):
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.warning(f"register_face: No frame available for sample {i+1}")
                continue

            encoding = deps.face_recognizer.get_single_face_encoding(frame)
            if encoding is not None:
                encodings.append(encoding)
                logger.debug(f"register_face: Got sample {len(encodings)}/{num_samples}")
            else:
                logger.debug(f"register_face: No face in sample {i+1}")

            if i < num_samples - 1:
                time.sleep(sample_delay)

        if len(encodings) == 0:
            logger.warning("register_face: No faces detected in any sample")
            return {"error": "No face detected. Please ensure you are looking at the camera with good lighting."}

        if len(encodings) < 3:
            logger.warning(f"register_face: Only got {len(encodings)} samples, quality may be lower")

        # Average the encodings for a more robust representation
        avg_encoding = np.mean(encodings, axis=0)
        logger.debug(f"register_face: Averaged {len(encodings)} encodings")

        # Store in database
        result = deps.face_db.add_face(name, avg_encoding)
        if "error" in result:
            logger.error(f"register_face: Database error: {result['error']}")
            return result

        logger.info(f"register_face: Successfully registered face for {name} using {len(encodings)} samples")
        return {
            "status": "success",
            "message": f"Successfully registered face for {name} (used {len(encodings)} samples for better accuracy)",
            "action": result.get("action", "added"),
            "samples_used": len(encodings),
        }


class RegisterFaceFromPhoto(Tool):
    """Register a face from a photo file."""

    name = "register_face_from_photo"
    description = (
        "Register a person's face from a photo file (JPG, PNG, etc). "
        "Use this when someone wants to register a face using an existing photo. "
        "Photos should be placed in the 'faces' folder in the project directory. "
        "You can provide just the filename (e.g., 'john.jpg') or a full path."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the person to register",
            },
            "photo_path": {
                "type": "string",
                "description": "The filename (e.g., 'john.jpg') or full path to the photo",
            },
        },
        "required": ["name", "photo_path"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Register a face from a photo file."""
        name = (kwargs.get("name") or "").strip()
        photo_path = (kwargs.get("photo_path") or "").strip()

        if not name:
            return {"error": "Name must be provided"}
        if not photo_path:
            return {"error": "Photo path must be provided"}

        logger.info(f"Tool call: register_face_from_photo name={name} photo_path={photo_path}")

        # Check dependencies
        if deps.face_db is None:
            logger.error("register_face_from_photo: Face database not initialized")
            return {"error": "Face database not initialized"}

        if deps.face_recognizer is None:
            logger.error("register_face_from_photo: Face recognizer not available")
            return {"error": "Face recognition not available"}

        # Resolve the photo path
        # 1. If it's just a filename, look in the faces folder
        # 2. If it starts with ~ or /, treat as full path
        if not os.path.isabs(photo_path) and not photo_path.startswith("~"):
            # Just a filename - look in faces folder
            resolved_path = os.path.join(FACES_FOLDER, photo_path)
        else:
            # Full path or ~ path
            resolved_path = os.path.expanduser(photo_path)

        logger.debug(f"Resolved photo path: {resolved_path}")

        # Check if file exists
        if not os.path.isfile(resolved_path):
            # Try to give a helpful error message
            if not os.path.isabs(photo_path) and not photo_path.startswith("~"):
                logger.error(f"register_face_from_photo: File not found in faces folder: {resolved_path}")
                return {"error": f"Photo not found. Please place '{photo_path}' in the faces folder: {FACES_FOLDER}"}
            else:
                logger.error(f"register_face_from_photo: File not found: {resolved_path}")
                return {"error": f"Photo file not found: {resolved_path}"}

        # Load the image
        frame = None
        try:
            if HAS_CV2:
                # OpenCV reads as BGR, which we'll convert in get_single_face_encoding
                frame = cv2.imread(resolved_path)
                if frame is not None:
                    # Convert BGR to RGB for face_recognition
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Loaded image with OpenCV: {frame.shape}")
            elif HAS_PIL:
                # PIL loads as RGB
                pil_image = Image.open(resolved_path)
                frame = np.array(pil_image.convert("RGB"))
                logger.debug(f"Loaded image with PIL: {frame.shape}")
            else:
                return {"error": "No image loading library available (need opencv-python or Pillow)"}
        except Exception as e:
            logger.error(f"register_face_from_photo: Error loading image: {e}")
            return {"error": f"Failed to load image: {e}"}

        if frame is None:
            return {"error": "Failed to load image file"}

        # Get face encoding from the photo
        # Note: face_recognizer expects BGR, so convert RGB back to BGR
        frame_bgr = frame[:, :, ::-1].copy()
        encoding = deps.face_recognizer.get_single_face_encoding(frame_bgr)

        if encoding is None:
            logger.warning(f"register_face_from_photo: No face detected in photo: {resolved_path}")
            return {"error": "No face detected in the photo. Please use a clear photo with a visible face."}

        # Store in database
        result = deps.face_db.add_face(name, encoding)
        if "error" in result:
            logger.error(f"register_face_from_photo: Database error: {result['error']}")
            return result

        logger.info(f"register_face_from_photo: Successfully registered face for {name} from {resolved_path}")
        return {
            "status": "success",
            "message": f"Successfully registered face for {name} from photo",
            "action": result.get("action", "added"),
            "photo": resolved_path,
        }


class IdentifyFace(Tool):
    """Identify who is currently visible to the camera."""

    name = "identify_face"
    description = (
        "Identify the person currently in view by comparing their face to registered faces. "
        "Use this when you want to know who you're talking to or to greet someone by name."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Identify faces in the current camera frame."""
        logger.info("Tool call: identify_face")

        # Check dependencies
        if deps.camera_worker is None:
            return {"error": "Camera not available"}

        if deps.face_db is None:
            return {"error": "Face database not initialized"}

        if deps.face_recognizer is None:
            return {"error": "Face recognition not available"}

        # Get current frame
        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"error": "No camera frame available"}

        # Get face encoding from frame
        encoding = deps.face_recognizer.get_single_face_encoding(frame)
        if encoding is None:
            return {"error": "No face detected in frame"}

        # Try to match against database
        match = deps.face_db.find_matching_face(encoding)

        if match is None:
            return {
                "identified": False,
                "message": "Face detected but not recognized. This person is not registered.",
            }

        name, distance = match
        # Confidence formula based on face_recognition library norms:
        # - distance < 0.4 is typically a good match
        # - distance < 0.6 is the threshold
        # Map: 0.0 -> 100%, 0.3 -> 85%, 0.4 -> 75%, 0.6 -> 50%
        if distance <= 0.3:
            # Excellent match: 85-100%
            confidence = int(100 - (distance / 0.3) * 15)
        elif distance <= 0.4:
            # Good match: 75-85%
            confidence = int(85 - ((distance - 0.3) / 0.1) * 10)
        else:
            # Acceptable match: 50-75%
            confidence = int(75 - ((distance - 0.4) / 0.2) * 25)

        confidence = max(50, min(100, confidence))  # Clamp to 50-100% for matches

        # Log the actual distance for debugging
        logger.debug(f"identify_face: Matched {name} with distance={distance:.3f}, confidence={confidence}%")

        return {
            "identified": True,
            "name": name,
            "confidence": confidence,
            "message": f"This is {name} (confidence: {confidence}%)",
        }


class ListKnownFaces(Tool):
    """List all registered faces in the database."""

    name = "list_known_faces"
    description = (
        "List all people whose faces have been registered. "
        "Use this to see who the robot can recognize."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """List all registered faces."""
        logger.info("Tool call: list_known_faces")

        if deps.face_db is None:
            return {"error": "Face database not initialized"}

        faces = deps.face_db.list_faces()
        count = len(faces)

        if count == 0:
            return {
                "count": 0,
                "message": "No faces registered yet.",
                "faces": [],
            }

        names = [f["name"] for f in faces]
        return {
            "count": count,
            "message": f"I know {count} {'person' if count == 1 else 'people'}: {', '.join(names)}",
            "faces": faces,
        }


class ForgetFace(Tool):
    """Remove a registered face from the database."""

    name = "forget_face"
    description = (
        "Remove a person's face from the database so they will no longer be recognized. "
        "Use this when asked to forget someone."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the person to forget",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Remove a face from the database."""
        name = (kwargs.get("name") or "").strip()
        if not name:
            return {"error": "Name must be provided"}

        logger.info(f"Tool call: forget_face name={name}")

        if deps.face_db is None:
            return {"error": "Face database not initialized"}

        result = deps.face_db.delete_face(name)
        if "error" in result:
            return result

        return {
            "status": "success",
            "message": f"I have forgotten {name}",
        }
