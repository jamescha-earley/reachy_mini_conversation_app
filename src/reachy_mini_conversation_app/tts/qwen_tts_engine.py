"""TTS engine that calls the Snowflake SPCS TTS service.

This engine sends text to a remote TTS service running on Snowpark Container Services,
which handles voice cloning with Qwen3-TTS on GPU compute.
"""

import base64
import logging
import asyncio
import os
from typing import AsyncGenerator, Optional

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

# Output sample rate to match OpenAI Realtime API
OUTPUT_SAMPLE_RATE = 24000


class Qwen3TTSEngine:
    """TTS Engine that calls remote SPCS service for voice cloning.

    This engine sends text to a Snowflake SPCS service running Qwen3-TTS,
    avoiding local dependency conflicts and GPU requirements.
    """

    def __init__(self, voice_sample_path: Optional[str] = None):
        """Initialize the TTS engine.

        Args:
            voice_sample_path: Path to the voice sample WAV file for cloning.
                               Will be uploaded to the service on first use.

        """
        self._voice_sample_path = voice_sample_path
        self._voice_id = "cartoon"
        self._voice_registered = False
        self._lock = asyncio.Lock()

        # Service endpoint - set via environment or config
        self._service_url = os.getenv("TTS_SERVICE_URL", "http://localhost:8000")
        
        # Snowflake connection for SPCS auth (optional)
        self._snowflake_connection = os.getenv("TTS_SNOWFLAKE_CONNECTION")
        self._token = None
        self._token_expiry = 0

        # HTTP client
        self._client = None

    def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def _get_auth_headers(self) -> dict:
        """Get authentication headers for SPCS endpoint using JWT key pair auth."""
        import time
        
        # If not using Snowflake auth, return empty headers (for local testing)
        if not self._snowflake_connection:
            return {}
        
        # Check if we need to refresh the token
        if self._token and time.time() < self._token_expiry - 60:
            return {"Authorization": f"Snowflake Token=\"{self._token}\""}
        
        try:
            import jwt
            import hashlib
            import toml
            import httpx
            import base64
            from pathlib import Path
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            
            # Load connection config from ~/.snowflake/connections.toml
            config_path = Path.home() / ".snowflake" / "connections.toml"
            if config_path.exists():
                config = toml.load(config_path)
                connection_params = config.get(self._snowflake_connection, {})
            else:
                logger.warning(f"Snowflake config not found at {config_path}")
                return {}
            
            if not connection_params:
                logger.warning(f"Connection '{self._snowflake_connection}' not found in config")
                return {}
            
            account_name = connection_params.get("account", "").replace("_", "-").lower()
            user = connection_params.get("user", "").upper()
            
            # Load private key
            key_path = Path.home() / ".snowflake" / "rsa_key.p8"
            if not key_path.exists():
                logger.warning(f"Private key not found at {key_path}")
                return {}
            
            with open(key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            
            # Get public key fingerprint
            public_key = private_key.public_key()
            public_key_der = public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            sha256_hash = hashlib.sha256(public_key_der).digest()
            fingerprint = "SHA256:" + base64.b64encode(sha256_hash).decode("utf-8")
            
            # First, get the account locator via a simple connection
            # We need to use the connector to get the account locator
            import snowflake.connector
            conn = snowflake.connector.connect(**connection_params)
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_ACCOUNT()")
            account_locator = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            # Create JWT with account locator
            now = int(time.time())
            qualified_username = f"{account_locator}.{user}"
            
            payload = {
                "iss": f"{qualified_username}.{fingerprint}",
                "sub": qualified_username,
                "iat": now,
                "exp": now + 3600,
            }
            
            jwt_token = jwt.encode(payload, private_key, algorithm="RS256")
            
            # Exchange JWT for session token
            login_endpoint = f"https://{account_name}.snowflakecomputing.com/session/v1/login-request"
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    login_endpoint,
                    json={
                        "data": {
                            "CLIENT_APP_ID": "TTS_CLIENT",
                            "CLIENT_APP_VERSION": "1.0.0",
                            "ACCOUNT_NAME": account_locator,
                            "LOGIN_NAME": user,
                            "AUTHENTICATOR": "SNOWFLAKE_JWT",
                            "TOKEN": jwt_token,
                        }
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                
                if response.status_code == 200:
                    resp_data = response.json()
                    if resp_data.get("success"):
                        self._token = resp_data["data"]["token"]
                        self._token_expiry = now + 3600
                        logger.info("Obtained Snowflake session token via JWT")
                        return {"Authorization": f"Snowflake Token=\"{self._token}\""}
                    else:
                        logger.warning(f"JWT auth failed: {resp_data.get('message')}")
                else:
                    logger.warning(f"JWT auth request failed: {response.status_code}")
            
        except Exception as e:
            logger.warning(f"Failed to get SPCS auth token: {e}")
            import traceback
            traceback.print_exc()
        
        return {}

    def set_voice_sample(self, path: str) -> None:
        """Update the voice sample path for cloning.

        Args:
            path: Path to the voice sample WAV file.

        """
        self._voice_sample_path = path
        self._voice_registered = False  # Need to re-register
        logger.info(f"Voice sample updated: {path}")

    async def _register_voice(self) -> None:
        """Upload voice sample to the TTS service."""
        if self._voice_registered or not self._voice_sample_path:
            return

        from pathlib import Path
        voice_path = Path(self._voice_sample_path)

        if not voice_path.exists():
            logger.warning(f"Voice sample not found: {self._voice_sample_path}")
            return

        logger.info(f"Registering voice sample with TTS service: {voice_path}")

        try:
            # Read and encode voice sample
            with open(voice_path, "rb") as f:
                audio_bytes = f.read()

            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Generate a unique voice ID from the file path
            self._voice_id = voice_path.stem

            # Register with the service
            client = self._get_client()
            headers = await self._get_auth_headers()
            response = await client.post(
                f"{self._service_url}/register_voice",
                json={
                    "voice_id": self._voice_id,
                    "audio_base64": audio_base64,
                },
                headers=headers,
            )
            response.raise_for_status()

            self._voice_registered = True
            logger.info(f"Voice registered successfully: {self._voice_id}")

        except Exception as e:
            logger.error(f"Failed to register voice: {e}")
            # Continue without voice cloning
            self._voice_id = "cartoon"

    async def generate_async(self, text: str) -> tuple[NDArray[np.float32], int]:
        """Generate speech from text using the TTS service.

        Args:
            text: The text to convert to speech.

        Returns:
            Tuple of (audio_array, sample_rate).

        """
        async with self._lock:
            # Ensure voice is registered
            await self._register_voice()

            logger.debug(f"Requesting TTS for: {text[:50]}...")

            try:
                client = self._get_client()
                headers = await self._get_auth_headers()
                response = await client.post(
                    f"{self._service_url}/synthesize",
                    json={
                        "text": text,
                        "voice_id": self._voice_id,
                        "temperature": 0.3,
                        "top_p": 0.95,
                    },
                    headers=headers,
                )
                response.raise_for_status()

                data = response.json()
                audio_base64 = data["audio_base64"]
                sample_rate = data["sample_rate"]

                # Decode audio
                audio_bytes = base64.b64decode(audio_base64)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio = audio_int16.astype(np.float32) / 32767.0

                logger.debug(f"Received {len(audio) / sample_rate:.2f}s of audio")
                return audio, sample_rate

            except Exception as e:
                logger.error(f"TTS service request failed: {e}")
                raise

    def generate(self, text: str) -> tuple[NDArray[np.float32], int]:
        """Synchronous wrapper for generate_async."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.generate_async(text)
                )
                return future.result()
        else:
            return loop.run_until_complete(self.generate_async(text))

    async def generate_streaming(
        self,
        text: str,
        chunk_size_ms: int = 100,
    ) -> AsyncGenerator[tuple[NDArray[np.int16], int], None]:
        """Generate speech and yield in chunks for playback.

        Note: Uses non-streaming TTS for better quality, then chunks locally.

        Args:
            text: The text to convert to speech.
            chunk_size_ms: Size of each audio chunk in milliseconds.

        Yields:
            Tuples of (audio_chunk_int16, sample_rate).

        """
        # Get full audio from service (better quality than streaming)
        audio, sample_rate = await self.generate_async(text)

        # Convert to int16 for compatibility
        audio_int16 = (audio * 32767).astype(np.int16)

        # Calculate chunk size in samples
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)

        # Yield chunks
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i : i + chunk_samples]
            yield chunk, sample_rate
            await asyncio.sleep(0.01)

    def is_available(self) -> bool:
        """Check if the TTS service is reachable."""
        try:
            import httpx
            response = httpx.get(f"{self._service_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
