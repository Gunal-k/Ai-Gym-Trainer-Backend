# **AI Gym Trainer** 🏋️‍♂️

The AI Gym Trainer is a full-stack, cross-platform mobile application designed to act as a personal fitness coach. It leverages real-time pose estimation for form correction, a conversational AI for workout and diet planning, and a complete user management system.

This project is divided into two main parts:

1.  **Frontend**: A mobile application built with React Native and Expo.
2.  **Backend**: An AI microservice built with Python and FastAPI.

-----

## **I. Frontend (Mobile App)**

The frontend provides the user interface for all app features, including user accounts, profile management, the AI chatbot, and the live camera for workout analysis.

### **Features**

  * User registration and login (Firebase Authentication).
  * User profile management with image uploads (Cloudinary).
  * Conversational AI chatbot for fitness and diet advice.
  * Live camera view with real-time pose analysis and audio feedback.

### **Technology Stack** ⚛️

  * **Framework**: React Native
  * **Platform**: Expo SDK
  * **Navigation**: React Navigation
  * **Cloud Services**: Firebase (Authentication, Firestore), Cloudinary (Image Storage)
  * **Camera & Media**: `react-native-vision-camera`, `expo-image-picker`, `expo-speech`

### **Setup & Running**

1.  **Prerequisites**:

      * Node.js (LTS version)
      * Java JDK 17
      * Android Studio with the Android SDK configured.

2.  **Clone & Install**:

    ```bash
    git clone <your-repository-url>
    cd <frontend-folder-name>
    npm install
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in the root of the frontend folder and add your credentials:

    ```env
    # Firebase Credentials
    EXPO_PUBLIC_FIREBASE_API_KEY="AIzaSy..."
    EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN="your-project.firebaseapp.com"
    EXPO_PUBLIC_FIREBASE_PROJECT_ID="your-project-id"
    # ... all other Firebase keys

    # Cloudinary Credentials
    EXPO_PUBLIC_CLOUDINARY_CLOUD_NAME="your_cloud_name"
    EXPO_PUBLIC_CLOUDINARY_UPLOAD_PRESET="your_upload_preset"
    ```

4.  **Run the App**:
    This project uses native libraries and **must be run as a development build**.

    ```bash
    # Ensure your backend is running first!
    # Connect a physical device or start an emulator.
    npx expo run:android
    ```

    After the initial build, any changes to your JavaScript code will reload instantly in the app.

-----

## **II. Backend (AI Server)**

The Python backend serves as the "brain" of the application, handling all complex AI and machine learning tasks.

### **Features**

  * **Pose Estimation Engine**: Analyzes video frames to detect exercises, count reps, and provide corrective feedback.
  * **Conversational AI**: Connects to the Google Gemini API to provide intelligent chat responses.
  * **Static Analysis**: Analyzes a single image to suggest workout plans.

### **Technology Stack** 🐍

  * **Framework**: FastAPI
  * **Server**: Uvicorn
  * **AI/ML**: Google MediaPipe (Pose Estimation), Google Gemini API (LLM)
  * **Environment**: Python 3.10 / 3.11

### **Setup & Running**

1.  **Prerequisites**:

      * Python (version 3.10 or 3.11).
      * A Google API Key for the Gemini API.

2.  **Clone & Install**:

    ```bash
    git clone <your-repository-url>
    cd <backend-folder-name>

    # Create and activate a virtual environment
    python -m venv venv
    .\venv\Scripts\activate

    # Install dependencies from the requirements file
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in the root of the backend folder and add your API key:

    ```env
    GOOGLE_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"
    ```

4.  **Run the Server**:
    The `--host 0.0.0.0` flag is critical for allowing the mobile app to connect.

    ```bash
    uvicorn main:app --reload --host 0.0.0.0
    ```

    The server will be available on your local network at `http://<your-computer-ip>:8000`.
