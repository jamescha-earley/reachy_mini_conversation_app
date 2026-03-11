#!/bin/bash
# =============================================================================
# Deploy TTS Service to Snowpark Container Services
# =============================================================================
# Usage: ./deploy.sh
# 
# Prerequisites:
# - Docker installed and running
# - Snowflake CLI installed (snow)
# - Authenticated to Snowflake: snow connection test -c pdevrel
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_REGISTRY="sfdevrel-devrel.registry.snowflakecomputing.com/tts_service_db/tts_schema/tts_images"
IMAGE_NAME="tts_service"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "TTS Service SPCS Deployment"
echo "=========================================="

# Step 1: Build Docker image
echo ""
echo "[1/4] Building Docker image..."
cd "$SCRIPT_DIR"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

# Step 2: Tag for Snowflake registry
echo ""
echo "[2/4] Tagging image for Snowflake registry..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"

# Step 3: Authenticate to Snowflake registry
echo ""
echo "[3/4] Authenticating to Snowflake registry..."
# Get auth token using Snowflake CLI
snow spcs image-registry token --connection pdevrel | docker login "${IMAGE_REGISTRY%%/*}" --username 0sessiontoken --password-stdin

# Step 4: Push to Snowflake registry
echo ""
echo "[4/4] Pushing image to Snowflake..."
docker push "${FULL_IMAGE}"

echo ""
echo "=========================================="
echo "Image pushed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the SQL to create the service (see below)"
echo "2. Wait for service to start (~2-3 minutes)"
echo "3. Get the service URL and update your .env"
echo ""
echo "Run this SQL in Snowflake:"
echo "----------------------------------------"
cat << 'EOF'
USE DATABASE TTS_SERVICE_DB;
USE SCHEMA TTS_SCHEMA;

CREATE SERVICE IF NOT EXISTS TTS_VOICE_CLONE_SERVICE
  IN COMPUTE POOL SYSTEM_COMPUTE_POOL_GPU
  FROM SPECIFICATION $$
spec:
  containers:
    - name: tts-service
      image: sfdevrel-devrel.registry.snowflakecomputing.com/tts_service_db/tts_schema/tts_images/tts_service:latest
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

-- Check status
DESCRIBE SERVICE TTS_VOICE_CLONE_SERVICE;
SHOW ENDPOINTS IN SERVICE TTS_VOICE_CLONE_SERVICE;
EOF
echo "----------------------------------------"
