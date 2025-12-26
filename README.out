# BeardAR - AI-Powered Beard Segmentation and AR Overlay System

A comprehensive system for capturing, segmenting, and projecting beard templates using Segment Anything Model (SAM) fine-tuned for beard detection, with real-time AR projection via MediaPipe Face Mesh.

## Architecture Overview

- **Frontend**: React + TypeScript + Vite, using MediaPipe Face Mesh for face tracking
- **Backend**: FastAPI (Python) with SAM model inference on MPS (Metal Performance Shaders) for M3 Mac
- **AI Model**: Segment Anything Model (SAM) - vit_b (Base) fine-tuned on beard dataset

---

## Directory Structure

### Root Files

- **README.md** - This file, comprehensive documentation of the entire project
- **visualize_beard_overlays.py** - Script to visualize beard segmentation overlays on test images

### `/backend` - Python FastAPI Backend

#### Core Application
- **main.py** - FastAPI application entry point, defines all API endpoints (health, segmentation, template creation/projection, frame storage)
- **requirements.txt** - Python dependencies (fastapi, torch, segment-anything, opencv-python, numpy, etc.)

#### `/backend/api` - API Layer
- **schemas.py** - Pydantic models for request/response validation (SegmentationRequest, BeardTemplate, etc.)
- **segmentation.py** - Core SAM segmentation logic, processes images with point prompts and returns masks
- **live_segmentation.py** - Live SAM segmentation using prompts derived from face mesh landmarks
- **beard_template.py** - BeardTemplateBuilder class that accumulates SAM masks from multiple frames, aligns them to canonical face space, and creates averaged templates
- **frame_storage.py** - Utilities to save/load captured frames to disk (JSON format) for reprocessing without rescanning

#### `/backend/core` - Core Services
- **model_loader.py** - Singleton ModelLoader class that loads SAM model onto MPS device, handles checkpoint loading (base + fine-tuned)

#### `/backend/train` - Training Scripts
- **dataset.py** - BeardCOCODataset class for loading COCO-formatted beard segmentation data, handles image padding/resizing for batching
- **train_sam.py** - Training script for fine-tuning SAM on beard dataset using MPS acceleration

#### `/backend/tests` - Unit Tests
- **test_segmentation.py** - Tests for segmentation endpoint and vertex containment logic
- **test_coordinates.py** - Tests for coordinate conversion utilities

#### Utility Scripts
- **test_sam_segmentation.py** - Script to test SAM segmentation on predefined test images and visualize results
- **test_my_images.py** - Script to test SAM segmentation on user-provided images in `my/` folder
- **create_beard_map.py** - Legacy script for creating 2.5D beard map by mapping SAM masks to face mesh vertices (deprecated approach)
- **create_beard_map_v2.py** - Proof-of-concept script for "Direct SAM" approach using template-derived prompts

### `/frontend` - React TypeScript Frontend

#### Configuration
- **package.json** - Frontend dependencies (React, TypeScript, Vite, Framer Motion, Zustand, Tailwind CSS)
- **vite.config.ts** - Vite build configuration, proxies `/api` requests to backend port 8000
- **tsconfig.json** - TypeScript compiler configuration
- **tailwind.config.js** - Tailwind CSS theme configuration (dark mode colors, animations)
- **postcss.config.js** - PostCSS configuration for Tailwind
- **index.html** - Main HTML file, loads MediaPipe Face Mesh and Camera Utils from CDN

#### Source Code (`/frontend/src`)
- **main.tsx** - React application entry point, renders App component
- **App.tsx** - Main application component, handles routing between Home/Scan/Project views
- **index.css** - Global CSS styles, dark mode theme variables, custom animations

#### `/frontend/src/components` - React Components
- **FaceTracker.tsx** - MediaPipe Face Mesh integration, handles webcam capture and 468-point face landmark detection
- **ScanView.tsx** - Face ID-like scanning interface, guides user through head rotations, captures frames and sends to backend for SAM segmentation
- **ProjectView.tsx** - Real-time AR projection view, loads saved template and projects beard outline onto live face using warped mask contour
- **CalibrationView.tsx** - Legacy calibration component (deprecated, replaced by ScanView)
- **ProjectionView.tsx** - Legacy projection component (deprecated, replaced by ProjectView)
- **index.ts** - Component exports

#### `/frontend/src/store` - State Management
- **beardStore.ts** - Zustand store for global state (saved templates, current landmarks), persists to localStorage

#### `/frontend/src/utils` - Utilities
- **api.ts** - API client functions for backend communication (segmentBeard, createBeardTemplateAPI, projectBeardTemplateAPI, listSavedScans, loadScanFrames)
- **coordinates.ts** - Coordinate conversion utilities, Catmull-Rom splines for smooth contour drawing
- **vite-env.d.ts** - TypeScript declarations for global MediaPipe types loaded from CDN

### `/scripts` - Shell Scripts
- **run_backend.sh** - Activates Python virtual environment and starts FastAPI backend server
- **run_frontend.sh** - Navigates to frontend directory, installs dependencies, and starts Vite dev server

### `/checkpoints` - Model Checkpoints
- **sam_vit_b_01ec64.pth** - Base SAM model checkpoint (downloaded from Meta)
- **sam_beard_best.pth** - Fine-tuned SAM model checkpoint (best validation IoU)
- **sam_beard_epoch_*.pth** - Intermediate training checkpoints (epochs 5, 10, 15, 20, 25, 30)
- **training_history.json** - Training metrics (loss, IoU) per epoch

### `/beard-dataset.v46i.coco-segmentation` - Training Dataset
- **README.dataset.txt** - Dataset information from Roboflow
- **train/** - Training images (139 JPG) and annotations (_annotations.coco.json)
- **valid/** - Validation images (55 JPG) and annotations
- **test/** - Test images (55 JPG) and annotations

### `/beard_map_results` - Legacy Results
- **beard_map.json** - Legacy beard map data (vertex indices approach)
- **beard_map_visualization.png** - Visualization of legacy beard map
- **test_comparison.png** - Comparison image from legacy approach

### `/beard_map_v2_results` - Direct SAM Results
- **aggregated_mask.png** - Aggregated mask from direct SAM approach
- **comparison.png** - Comparison visualization
- **final_result.png** - Final result from proof-of-concept

### `/my` - User Test Images
- **Screenshot *.png** - User-provided test images for SAM segmentation
- **my_test/test.png** - Additional test image

### `/my_results` - User Test Results
- **result_*.png** - SAM segmentation results on user test images

### `/scans` - Saved Scan Data
- **{template_id}_frames.json** - Saved frame data (images, landmarks, SAM masks) for reprocessing without rescanning

### `/segment-anything` - SAM Library (Submodule/Clone)
- Facebook Research's Segment Anything Model implementation
- **segment_anything/** - Core SAM model code (image encoder, mask decoder, prompt encoder, transformer)
- **notebooks/** - Example Jupyter notebooks for SAM usage
- **demo/** - Web demo application (not used in this project)

---

## Key Workflows

### 1. Training SAM on Beard Dataset
1. Place COCO dataset in `beard-dataset.v46i.coco-segmentation/`
2. Run `backend/train/train_sam.py` to fine-tune SAM
3. Best checkpoint saved to `checkpoints/sam_beard_best.pth`

### 2. Scanning Face (Template Creation)
1. User navigates to Scan view
2. FaceTracker captures webcam feed and detects 468 face landmarks
3. ScanView guides user through head rotations (center, left, right, up, down)
4. For each frame: image + landmarks sent to `/api/template/add-frame`
5. Backend runs SAM segmentation, accumulates masks in canonical face space
6. Frames automatically saved to `scans/{template_id}_frames.json`
7. After 60 frames: `/api/template/finalize` averages masks, creates template

### 3. Projecting Template (AR Overlay)
1. User navigates to Project view
2. FaceTracker provides live face landmarks
3. ProjectView calls `/api/template/{id}/project` with current landmarks
4. Backend warps pre-computed template mask to live face pose
5. Frontend draws projected contour on canvas overlay

### 4. Loading Saved Scans (Reprocessing)
1. User sees list of saved scans on home page
2. Click "Load" to reprocess saved frames without rescanning
3. Backend loads frames from JSON, reruns SAM if needed, creates template
4. Useful for testing different thresholds/parameters

---

## API Endpoints

### Health
- `GET /` - Root health check
- `GET /health` - Detailed health check (model status, device)

### Segmentation
- `POST /segment` - Run SAM segmentation with point prompts
- `POST /segment/live` - Live SAM segmentation using face landmarks

### Template Management
- `POST /template/add-frame` - Add a frame to template builder (called during scan)
- `POST /template/finalize` - Finalize template after all frames collected
- `POST /template/project` - Project saved template onto live face
- `DELETE /template/{template_id}` - Delete a template

### Frame Storage
- `GET /scans` - List all saved scans
- `POST /scans/{template_id}/load` - Load and reprocess saved frames
- `DELETE /scans/{template_id}` - Delete a saved scan

---

## Technology Stack

- **Backend**: Python 3.14, FastAPI, PyTorch (MPS), OpenCV, NumPy
- **Frontend**: React 18, TypeScript, Vite, Framer Motion, Zustand, Tailwind CSS
- **AI/ML**: Segment Anything Model (SAM), MediaPipe Face Mesh
- **Build Tools**: Vite, PostCSS, Tailwind CSS
- **State Management**: Zustand (with localStorage persistence)

---

## Development Setup

1. **Backend Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   cd backend
   pip install -r requirements.txt
   # Download SAM checkpoint to checkpoints/sam_vit_b_01ec64.pth
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   ```

3. **Run Services**:
   ```bash
   # Terminal 1: Backend
   ./scripts/run_backend.sh
   
   # Terminal 2: Frontend
   ./scripts/run_frontend.sh
   ```

4. **Access**: http://localhost:3000

---

## Notes

- MediaPipe is loaded from CDN (not npm) due to Vite ES module compatibility issues
- Face landmarks are mirrored to match webcam feed
- Template masks are stored in canonical face space (512x512) for rotation invariance
- Frame data is saved automatically during scan for reprocessing
- All logging goes to stderr for debugging (check backend terminal for detailed logs)
