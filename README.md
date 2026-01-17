# AI Study Companion & Mood Tracker

An intelligent, real-time productivity dashboard that bridges **Computer Vision** and **Web Development** to help students optimize their focus. By analyzing facial expressions, the app tracks emotional states during study sessions and provides actionable insights to prevent burnout.

---

## Key Features

- **Real-time Emotion Analysis**: Captures live video frames via webcam and processes them using a custom CNN model to identify 7 distinct emotional states.
- **Live Productivity Scoring**: Converts emotional data into a numerical "Focus Score" to track study efficiency over time.
- **Dynamic Recommendations**: Provides AI-generated tips based on current mood (e.g., suggesting a break if stress is detected).
- **Data Visualization**: Includes a Mood Distribution Map and Session History Log for deep daily reflection.
- **Full-Stack Architecture**: Demonstrates seamless communication between a **TypeScript (Next.js)** frontend and a **Python (FastAPI)** backend.

---

## Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Lucide Icons
- **Backend**: Python 3.x, FastAPI, OpenCV, NumPy
- **AI/ML**: Deep Learning (CNN) for Facial Expression Recognition
- **State Management**: React Hooks (useState, useEffect, useRef)

---

## Architecture



The application follows a decoupled architecture:
1. **Frontend**: Captures screenshots every 3 seconds and sends them to the API via Axios.
2. **Backend**: Pre-processes images (grayscale, normalization) and runs inference using the trained model.
3. **Analytics**: The frontend calculates session statistics locally to provide instant visual feedback.

---
