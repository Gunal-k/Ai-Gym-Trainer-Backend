# AI Gym Trainer (Backend)

This is the Python backend microservice for the AI Gym Trainer application. It uses FastAPI to serve endpoints for real-time pose estimation, static image analysis, and a conversational AI chatbot.

---
## **Technology Stack** 🐍

* **Framework**: FastAPI
* **Server**: Uvicorn
* **AI / ML**:
    * **Pose Estimation**: Google MediaPipe
    * **Chatbot LLM**: Google Gemini API
* **Environment**: Python 3.8+

---
## **Prerequisites**

* Python 3.8 or newer
* `pip` (Python package installer)
* A Google API Key with the Gemini API enabled

---
## **Setup Instructions**

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <backend-folder-name>
    ```

2.  **Create and Activate a Virtual Environment**
    This isolates your project's dependencies.
    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:
    ```txt
    fastapi
    uvicorn
    python-multipart
    mediapipe
    opencv-python-headless
    google-generativeai
    python-dotenv
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**

    Create a file named `.env` in the root of the backend folder and add your secret API key.

    ```env
    # .env file
    GOOGLE_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"
    ```

---
## **Running the Server**

1.  **Start the Backend Server**
    With your virtual environment active, run the following command:
    ```bash
    uvicorn main:app --reload
    ```
    # To expose the server to your local network (e.g., for mobile app testing), use:
    # .\venv\Scripts\activate
    # uvicorn analysis:app --host 0.0.0.0 --port 8000 --reload
    # uvicorn chat_service:app --host 0.0.0.0 --port 8001 --reload
    
    The `--reload` flag automatically restarts the server when you make changes to the code.

2.  **Verify the Server**
    The server will be running on `http://127.0.0.1:8000`. You can open this address in your web browser to see the message `{"message":"AI Backend is running."}`.