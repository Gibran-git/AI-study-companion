import cv2
import numpy as np
import base64
import tf_keras as keras
from tf_keras.models import model_from_json
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. CORS Setup: Allows your Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites (for local development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Architecture from JSON and Weights from H5
try:
    # Load the blueprint
    with open("models/fer.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    
    # Load the knowledge into the blueprint
    model.load_weights("models/fer.h5")
    print("✅ SUCCESS: Model fully reconstructed from JSON and H5.")
except Exception as e:
    print(f"❌ ERROR: Architecture or Weight mismatch. {e}")

# 3. Detectors and Labels
# These labels must stay in this specific order to match the model's output
# Standard FER-2013 Label Order
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

@app.get("/")
def home():
    return {"status": "online", "message": "Backend is ready for Phase 2"}

# 4. The Image Analysis Endpoint
# Recommendation Dictionary
MOOD_TIPS = {
    "Neutral": "You're in the flow. Stay focused!",
    "Happy": "Great mood! Tackle your hardest task now.",
    "Sad": "Feeling a bit low? Try a 5-minute walk or a quick stretch. Tackle an easy task now.",
    "Angry": "Frustrated? do box breathing to reset. Simlify or switch tasks briefly",
    "Fear": "Anxious? Break your current task into smal steps. Focus on one step at a time.",
    "Disgust": "Environment feeling off? Take a 2-minute reset. Try changing task or study method",
    "Surprise": "Focused or startled? Refocus and continue." 
}

@app.post("/analyze")
async def analyze_frame(data: dict = Body(...)):
    image_data = data.get("image")
    if not image_data: 
        return {"error": "No image data found"}

    try:
        # --- 2. Base64 Decoding & Padding Fix ---
        format, imgstr = image_data.split(';base64,') 
        
        # Add '=' padding if the string length isn't a multiple of 4
        missing_padding = len(imgstr) % 4
        if missing_padding:
            imgstr += '=' * (4 - missing_padding)
            
        nparr = np.frombuffer(base64.b64decode(imgstr), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- 3. Face Processing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.equalizeHist(roi_gray) 
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float32") / 255.0
            
            # Reshape for Keras
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # --- 4. Prediction & Recommendation ---
            prediction = model.predict(roi_gray)
            max_index = int(np.argmax(prediction))
            label = EMOTIONS[max_index]
            
            # Match the label to the tip
            tip = MOOD_TIPS.get(label, "Keep up the good work!")
            
            return {
                "emotion": label,
                "recommendation": tip
            }

        return {"emotion": "No Face Detected", "recommendation": "Position yourself in the camera."}

    except Exception as e:
        # Print the specific error to your terminal for debugging
        print(f"Error: {str(e)}")
        return {"error": str(e)}
if __name__ == "__main__":
    import uvicorn
    # Start the server on localhost port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)