-- ============================================================================
-- SPCS Deployment Script for TTS Voice Cloning Service
-- ============================================================================
-- Run these commands in Snowflake to deploy the TTS service
-- Prerequisites: SPCS enabled, GPU compute pool available
-- ============================================================================

-- 1. Create database and schema for the service
CREATE DATABASE IF NOT EXISTS TTS_SERVICE_DB;
USE DATABASE TTS_SERVICE_DB;
CREATE SCHEMA IF NOT EXISTS TTS_SCHEMA;
USE SCHEMA TTS_SCHEMA;

-- 2. Create image repository
CREATE IMAGE REPOSITORY IF NOT EXISTS TTS_IMAGES;

-- Get the repository URL (you'll need this for docker push)
SHOW IMAGE REPOSITORIES;
-- Note the repository_url, it will look like:
-- <org>-<account>.registry.snowflakecomputing.com/tts_service_db/tts_schema/tts_images

-- 3. Create compute pool with GPU (adjust size as needed)
CREATE COMPUTE POOL IF NOT EXISTS TTS_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S  -- NVIDIA GPU instance
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300;    -- Suspend after 5 min idle to save costs

-- Check compute pool status
DESCRIBE COMPUTE POOL TTS_GPU_POOL;

-- 4. Create the service (run after pushing the Docker image)
-- Replace <IMAGE_REGISTRY> with your actual registry URL from step 2
CREATE SERVICE IF NOT EXISTS TTS_VOICE_CLONE_SERVICE
  IN COMPUTE POOL TTS_GPU_POOL
  FROM SPECIFICATION $$
spec:
  containers:
    - name: tts-service
      image: <IMAGE_REGISTRY>/tts_service:latest
      resources:
        requests:
          nvidia.com/gpu: 1
          memory: 16Gi
          cpu: 4
        limits:
          nvidia.com/gpu: 1
          memory: 32Gi
          cpu: 8
      env:
        NVIDIA_VISIBLE_DEVICES: all
  endpoints:
    - name: tts-api
      port: 8000
      public: true
$$
  MIN_INSTANCES = 1
  MAX_INSTANCES = 1;

-- 5. Check service status
SHOW SERVICES;
DESCRIBE SERVICE TTS_VOICE_CLONE_SERVICE;

-- 6. Get the service endpoint URL
SHOW ENDPOINTS IN SERVICE TTS_VOICE_CLONE_SERVICE;
-- The ingress_url is what you'll use in your app config

-- 7. View service logs (useful for debugging)
SELECT SYSTEM$GET_SERVICE_LOGS('TTS_VOICE_CLONE_SERVICE', 0, 'tts-service');

-- ============================================================================
-- Usage from Python (after deployment):
-- 
-- import httpx
-- 
-- TTS_ENDPOINT = "https://<your-service-url>/synthesize"
-- 
-- response = httpx.post(TTS_ENDPOINT, json={
--     "text": "Hello, this is my cloned voice!",
--     "voice_id": "my_voice"
-- })
-- audio_base64 = response.json()["audio_base64"]
-- ============================================================================

-- Cleanup commands (when you want to remove the service):
-- DROP SERVICE IF EXISTS TTS_VOICE_CLONE_SERVICE;
-- DROP COMPUTE POOL IF EXISTS TTS_GPU_POOL;
