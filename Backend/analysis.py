import cv2
import numpy as np
import mediapipe as mp
import base64
import random
import time
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feedback phrases
# --- Feedback Libraries ---
ENCOURAGEMENT_PHRASES = [
    "You can do it!", "Push through!", "Almost there!", "Don't give up!",
    "Great effort!", "Stay strong!", "Finish this rep!", "Focus!"
]
# Push-up Feedback
PUSHUP_DOWN_CUES = ["Lower.", "Chest towards floor.", "Elbows back."]
PUSHUP_UP_CUES = ["Push up.", "Extend fully.", "Power up!"]
PUSHUP_FORM_CORRECTIONS = ["Keep back straight.", "Don't drop your hips.", "Engage your core."]
# Squat Feedback
SQUAT_DOWN_CUES = ["Lower hips.", "Break parallel.", "Good depth!", "Sit back."]
SQUAT_UP_CUES = ["Drive up!", "Stand tall.", "Squeeze glutes."]
SQUAT_FORM_CORRECTIONS = ["Keep chest up.", "Knees out.", "Weight on heels."]
# Bicep Curl Feedback
BICEP_CURL_UP_CUES = ["Curl up.", "Squeeze.", "Elbows stable."]
BICEP_CURL_DOWN_CUES = ["Lower slowly.", "Control the weight.", "Full extension."]
BICEP_FORM_CORRECTIONS = ["Don't swing.", "Keep elbows tucked."]
# General Feedback
GOOD_FORM_PHRASES = ["Perfect form!", "Great job!", "Looking good!", "Keep it up!"]
VISIBILITY_CUES = ["Keep body visible.", "Step back slightly.", "Center yourself."]
STARTING_CUES = ["Get into starting pose.", "Ready?"]

VISIBILITY_THRESHOLD = 0.6


# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.5)

def get_landmark_coords(lm, landmark_enum, threshold=0.3):
    idx = landmark_enum.value
    if idx < len(lm) and lm[idx].visibility > threshold:
        return [lm[idx].x, lm[idx].y]
    return None

# --- Helpers ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Exercise analyzers
def analyze_pushup(landmarks, state):
    feedback = ""
    try:
        # --- 1. Get Landmarks Safely ---
        # Get coordinates for both sides. They will be None if not visible.
        left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        left_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        
        right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        right_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
        right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        right_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)

        # --- 2. Calculate Angles (only if landmarks are visible) ---
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_back_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # --- 3. Select Valid Angles (Handles Side vs. Front View) ---
        
        # For elbow angle (rep counting):
        # Use average if both arms are visible, otherwise use the one visible arm.
        if left_elbow_angle != -1.0 and right_elbow_angle != -1.0:
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        elif left_elbow_angle != -1.0:
            elbow_angle = left_elbow_angle
        elif right_elbow_angle != -1.0:
            elbow_angle = right_elbow_angle
        else:
            # If neither arm is visible, we can't do anything.
            return random.choice(VISIBILITY_CUES) + " (Arms)", state

        # For back angle (form check):
        # Only use an angle if it's calculable.
        if left_back_angle != -1.0 and right_back_angle != -1.0:
            back_angle = (left_back_angle + right_back_angle) / 2
        elif left_back_angle != -1.0:
            back_angle = left_back_angle
        elif right_back_angle != -1.0:
            back_angle = right_back_angle
        else:
            back_angle = -1.0 # Flag as not calculable (e.g., front-on view)

        # --- 4. Priority 1: Form Correction (Back Straightness) ---
        # Only check the back angle if it was calculable (i.e., not a front-on view)
        if back_angle != -1.0 and back_angle < 155:
            feedback = "Keep your back straight!"
            return feedback, state

        # --- 5. Priority 2: Rep Counting & Stage Feedback ---
        rep_counted_this_frame = False
        time_in_stage = time.time() - state["last_stage_time"]

        if elbow_angle > 160: # Arms extended (Up position)
            if state['stage'] == 'down':
                state['counter'] += 1
                feedback = f"Rep {state['counter']}"
                rep_counted_this_frame = True
            state['stage'] = 'up'
        elif elbow_angle < 90: # Arms bent (Down position)
            if state['stage'] != 'down':
                feedback = random.choice(PUSHUP_DOWN_CUES)
            state['stage'] = 'down'
            state['last_stage_time'] = time.time()
        else: # Mid-range
            if not rep_counted_this_frame:
                if state['stage'] == 'up': feedback = "Lower your chest."
                else: feedback = random.choice(PUSHUP_UP_CUES)

        # --- Priority 3: Encouragement (if struggling) ---
        if not feedback and time_in_stage > 4 and state["stage"] == 'down':
            feedback = random.choice(ENCOURAGEMENT_PHRASES)

        # --- Priority 4: General Good Form (if nothing else) ---
        if not feedback and random.random() < 0.1:
            feedback = random.choice(GOOD_FORM_PHRASES)

    except Exception as e:
        # print(f"Pushup analysis error: {e}") # Uncomment for debugging
        feedback = random.choice(VISIBILITY_CUES)

    return (feedback if feedback else "Keep going."), state
def analyze_bicep_curl(landmarks, state):
    """
    More robust bicep curl analyzer.
    Expects state to include left_counter,left_stage,right_counter,right_stage.
    """
    feedback = ""
    rep_counted_this_frame = False
    VISIBILITY_THRESHOLD = 0.6

    # Ensure state keys exist
    state.setdefault('left_counter', 0)
    state.setdefault('right_counter', 0)
    state.setdefault('left_stage', 'down')
    state.setdefault('right_stage', 'down')

    # thresholds
    DOWN_ANGLE = 160.0
    UP_ANGLE = 40.0
    HYSTERESIS_DOWN = 55.0  # if currently up, require angle > this to consider back to down

    try:
        # safe get coords (get_landmark_coords already checks visibility)
        left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow    = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist    = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)

        right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        right_elbow    = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist    = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

        left_angle = None
        right_angle = None
        if None not in (left_shoulder, left_elbow, left_wrist):
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if None not in (right_shoulder, right_elbow, right_wrist):
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Left arm logic
        if left_angle is not None:
            # mark down when straight
            if left_angle > DOWN_ANGLE:
                state['left_stage'] = 'down'
            # count upward curl if previously down
            elif left_angle < UP_ANGLE and state['left_stage'] == 'down':
                state['left_counter'] += 1
                state['left_stage'] = 'up'
                rep_counted_this_frame = True
                feedback = f"Rep {state['left_counter'] + state['right_counter']}"
            # if currently up but angle relaxed back beyond hysteresis, allow future rep
            elif state['left_stage'] == 'up' and left_angle > HYSTERESIS_DOWN:
                state['left_stage'] = 'down'

        # Right arm logic
        if right_angle is not None:
            if right_angle > DOWN_ANGLE:
                state['right_stage'] = 'down'
            elif right_angle < UP_ANGLE and state['right_stage'] == 'down':
                state['right_counter'] += 1
                state['right_stage'] = 'up'
                rep_counted_this_frame = True
                feedback = f"Rep {state['left_counter'] + state['right_counter']}"
            elif state['right_stage'] == 'up' and right_angle > HYSTERESIS_DOWN:
                state['right_stage'] = 'down'

        # corrective feedback if nothing counted
        if not rep_counted_this_frame and not feedback:
            # only give cues if we have angles to reason with
            if (state['left_stage'] == 'down' and left_angle is not None and 40 < left_angle < 150) or \
               (state['right_stage'] == 'down' and right_angle is not None and 40 < right_angle < 150):
                feedback = random.choice(BICEP_CURL_UP_CUES)
            elif (state['left_stage'] == 'up' and left_angle is not None and left_angle > 50) or \
                 (state['right_stage'] == 'up' and right_angle is not None and right_angle > 50):
                feedback = random.choice(BICEP_CURL_DOWN_CUES)

    except Exception as e:
        # return visibility cue on unexpected errors (optionally log e)
        feedback = random.choice(VISIBILITY_CUES)

    state['counter'] = state['left_counter'] + state['right_counter']
    return (feedback if feedback else "Keep curling."), state

def analyze_squat(landmarks, state):
    feedback = ""
    VISIBILITY_THRESHOLD = 0.6
    try:
        # --- Check Visibility ---
        left_hip_vis = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
        left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
        left_ankle_vis = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
        right_hip_vis = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
        right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
        right_ankle_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility
        # Add shoulder visibility checks for back angle
        left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility


        if not (left_hip_vis > VISIBILITY_THRESHOLD and left_knee_vis > VISIBILITY_THRESHOLD and left_ankle_vis > VISIBILITY_THRESHOLD and \
                right_hip_vis > VISIBILITY_THRESHOLD and right_knee_vis > VISIBILITY_THRESHOLD and right_ankle_vis > VISIBILITY_THRESHOLD and \
                left_shoulder_vis > VISIBILITY_THRESHOLD and right_shoulder_vis > VISIBILITY_THRESHOLD): # Check shoulders too
             return random.choice(VISIBILITY_CUES), state

        # --- Get Coordinates Safely ---
        left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
        left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

        if None in [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_shoulder, right_shoulder]:
             return random.choice(VISIBILITY_CUES), state

        # --- Calculate Angles ---
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        left_back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_back_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        avg_back_angle = (left_back_angle + right_back_angle) / 2

        # --- Priority 1: Form Correction (Back Angle) ---
        if avg_back_angle < 80: # Example threshold for excessive forward lean
             feedback = "Keep chest up!"
             return feedback, state

        # --- Priority 2: Rep Counting & Stage Feedback ---
        rep_counted_this_frame = False
        if avg_knee_angle > 165: # Standing up
            if state['stage'] == 'down':
                state['counter'] += 1
                feedback = f"Rep {state['counter']}"
                rep_counted_this_frame = True
            state['stage'] = 'up'
        elif avg_knee_angle < 90: # At the bottom of the squat
            if state['stage'] != 'down':
                feedback = random.choice(SQUAT_DOWN_CUES)
            state['stage'] = 'down'
        else: # Mid-range
            if not rep_counted_this_frame:
                if state['stage'] == 'up':
                    feedback = "Lower hips."
                else:
                    feedback = random.choice(SQUAT_UP_CUES)

        # --- Add more form corrections if needed (e.g., knee wobble) ---
        # knee_diff = abs(left_knee_angle - right_knee_angle)
        # if knee_diff > 15 and not rep_counted_this_frame and not feedback:
        #     feedback = "Keep knees aligned."

    except Exception as e:
        # print(f"Squat analysis error: {e}")
        feedback = random.choice(VISIBILITY_CUES)

    return (feedback if feedback else "Good form!"), state

# Starting pose classifier
def classify_pose(landmarks):
    """
    Analyzes body landmarks with stricter checks to differentiate standing/plank.
    """
    VISIBILITY_THRESHOLD = 0.6
    try:
        # --- Get Key Landmarks Safely ---
        left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        # Add others needed for calculations
        left_knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)

        # Early exit if core vertical landmarks are missing
        if None in [left_shoulder, left_hip, left_ankle, left_knee]:
            # print("Classifier: Missing core vertical landmarks.") # Debugging
            return None

        # --- Calculate Overall Body Orientation ---
        # Vertical distance (height) vs Horizontal distance (shoulder span approx.)
        body_height = abs(left_ankle[1] - left_shoulder[1]) # Y-difference
        # Use hip-shoulder X-difference as a proxy for width (more stable than shoulders alone sometimes)
        body_width_proxy = abs(left_hip[0] - left_shoulder[0]) 

        # --- Check 1: Is the pose predominantly VERTICAL? ---
        # If height is much greater than width, it's likely standing or sitting
        is_predominantly_vertical = body_height > body_width_proxy * 1.5 # Heuristic: Height > 1.5x Width

        if is_predominantly_vertical:
             # --- Now check if standing for Squat/Bicep Curl ---
             leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
             if leg_angle > 165: # Standing Straight
                 # Bicep Curl Check (arms down?)
                 if all(coord is not None for coord in [left_elbow, left_wrist]):
                     arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                     wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y # Use direct landmark for y comparison
                     hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                     if arm_angle > 155 and wrist_y > hip_y:
                         return "bicep_curl"
                 # Fallback: General Standing Pose = Squat
                 return "squat"
             else:
                  # Not fully straight, could be sitting or mid-squat - ignore for now
                  # Could add a sitting classification here if needed later
                  return None # Not a recognized starting pose if legs bent significantly

        # --- Check 2: If NOT predominantly vertical, check for PUSHUP (Plank) ---
        else:
             # Requires ankles, hips, shoulders to be visible
             if all(coord is not None for coord in [left_ankle, left_hip, left_shoulder]):
                 shoulder_y = left_shoulder[1]
                 hip_y = left_hip[1]
                 ankle_y = left_ankle[1]

                 # Check horizontal alignment (Y-coordinates relatively close)
                 # Using slightly larger threshold as horizontal check comes *after* vertical check
                 if abs(shoulder_y - hip_y) < 0.2 and abs(hip_y - ankle_y) < 0.2:
                      # Add leg straight check for pushup
                      leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
                      if leg_angle > 160: # Legs should be straight in a plank
                          return "pushup"

    except Exception as e:
        # print(f"Classifier error: {e}") # Uncomment for debugging
        return None

    return None # Default if no pose matched or error occurred

@app.websocket("/analysis/live-posture")
async def live_posture(ws: WebSocket):
    await ws.accept()
    print("WebSocket connection established.")
    state = {"counter":0, "stage":"","current_workout":None, "last_time":time.time(),
             "left_counter": 0,   # <-- ADD THIS
        "left_stage": "down",  # <-- ADD THIS
        "right_counter": 0,  # <-- ADD THIS
        "right_stage": "down"} # Use last_time if needed
    try:
        while True:
            # --- MODIFIED: Receive TEXT (Base64 string) ---
            base64_string = await ws.receive_text()

            # --- ADDED: Decode the Base64 string ---
            try:
                # Strip potential data URI prefix just in case
                if "," in base64_string:
                    base64_string = base64_string.split(',')[1]
                img_data = base64.b64decode(base64_string)
            except Exception as e:
                print(f"Error decoding Base64: {e}")
                continue # Skip this frame

            # Decode the image bytes using OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Could not decode image from bytes.")
                continue

            # --- The rest of your processing logic remains the same ---
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feedback = "Align yourself in the frame."

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if not state["current_workout"] or state["current_workout"] == "unknown":
                    detected_pose = classify_pose(lm)
                    if detected_pose:
                        detected_pose = detected_pose.replace(" ", "_")
                        state["current_workout"] = detected_pose
                        # Reset counters and stages for fresh session
                        state.update({
                            "counter": 0,
                            "stage": "",
                            "left_counter": 0,
                            "right_counter": 0,
                            "left_stage": "down",
                            "right_stage": "down",
                            "last_stage_time": time.time()
                        })
                        feedback = f"Detected {detected_pose}. Let's begin."

                    else:
                        feedback = "Get into starting pose."
                        state["current_workout"] = "unknown" # Stay in detection mode

                # --- Call analysis functions based on detected workout ---
                elif state["current_workout"] == "pushup":
                    feedback, state = analyze_pushup(lm, state)
                elif state["current_workout"]=="bicep_curl":
                    feedback,state = analyze_bicep_curl(lm,state)
                elif state["current_workout"] == "squat":
                    feedback, state = analyze_squat(lm, state)

            await ws.send_json({
                "feedback": feedback,
                "reps": state['counter'],
                "exercise": state.get("current_workout", "None")
            })

    except Exception as e:
        print(f"WebSocket closed with error: {e}")
    finally:
        print("WebSocket connection process finished.")
        # Optionally try closing gracefully if not already closed
        try:
             await ws.close()
        except RuntimeError:
             pass # Connection already closed

@app.post("/analyze/snapshot")
async def analyze_snapshot(file: UploadFile = File(...)):
    """
    Analyzes a single snapshot, detects pose, and returns all 33
    landmark coordinates in the format the frontend expects.
    """
    try:
        # 1. Read and decode the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # 2. Process the image for pose
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            raise HTTPException(status_code=400, detail="Could not detect a person in the image.")

        # 3. Format landmarks for the frontend
        # The JS code expects an array of objects with "id" and "x" keys
        landmarks_data = []
        all_landmarks = results.pose_landmarks.landmark
        
        for id_value, lm in enumerate(all_landmarks):
            landmarks_data.append({
                "id": id_value,
                "name": mp_pose.PoseLandmark(id_value).name,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        # 4. Return the data
        return {"landmarks": landmarks_data}

    except Exception as e:
        print(f"Snapshot Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.get("/")
def root(): return {"message":"Pose Analysis Service Running"}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)