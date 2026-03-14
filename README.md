# 🤖 Prisma-Gemini-DeepThink-API - Smart Multi-Expert Reasoning Pipeline

[![Download Prisma-Gemini-DeepThink-API](https://img.shields.io/badge/Download-Prisma--Gemini--DeepThink--API-brightgreen)](https://github.com/ramkoirala11235/Prisma-Gemini-DeepThink-API)

---

## 📋 What is Prisma-Gemini-DeepThink-API?

Prisma-Gemini-DeepThink-API is a backend system designed to handle complex questions by splitting them into smaller tasks. It uses many "experts" that work at the same time. After each expert gives its answers, the system reviews and refines those answers. Finally, it combines everything into one clear response.

This project works with models like Gemini 3.1 Pro and others. It targets tasks like creative writing, translations, and deep analysis—areas where thoughtful and subjective answers are needed.

This software only includes the backend part. It connects to a front-end called Prisma, but that part is not included here. Instead, this API uses Python and FastAPI to manage the reasoning process.

---

## 🚀 Getting Started

This section will help you download and run Prisma-Gemini-DeepThink-API on your Windows computer. You do not need coding skills to get it running.

### System Requirements

- Windows 10 or later
- At least 4 GB of RAM
- About 500 MB free disk space
- Python 3.9 or newer installed
- Internet connection to use AI models

---

## 📥 Download Prisma-Gemini-DeepThink-API

[![Download Here](https://img.shields.io/badge/Download-Now-blue?style=for-the-badge)](https://github.com/ramkoirala11235/Prisma-Gemini-DeepThink-API)

Visit the link above to download the necessary files. This link leads to the full project repository on GitHub, where you can get all the files you need.

---

## 🛠️ How to Install Python (if needed)

If you do not have Python installed:

1. Open your browser.
2. Go to https://www.python.org/downloads/windows/
3. Download the latest Python 3 version for Windows.
4. Run the installer.
5. During installation, check "Add Python to PATH" option.
6. Finish installation.

---

## ⚙️ Setup Instructions

These steps will guide you through setting up Prisma-Gemini-DeepThink-API on your Windows machine.

1. **Download the project files**

    - Go to [Prisma-Gemini-DeepThink-API GitHub page](https://github.com/ramkoirala11235/Prisma-Gemini-DeepThink-API).
    - Click the green "Code" button.
    - Select "Download ZIP."
    - Save and unzip the files in a folder you can access easily, for example, `C:\PrismaDeepThink\`.

2. **Open PowerShell or Command Prompt**

    - Press Windows key + R, type `cmd`, and press Enter.
    - Or, press Windows key, type “PowerShell,” and open it.

3. **Navigate to the folder**

    - In the command window, enter:
      
      ```bash
      cd C:\PrismaDeepThink
      ```

    - Replace `C:\PrismaDeepThink` with your actual folder path.

4. **Create a virtual environment**

    This step keeps the installation clean:
    
    ```bash
    python -m venv env
    ```

5. **Activate the virtual environment**

    - For Command Prompt:
      
      ```bash
      env\Scripts\activate
      ```
    
    - For PowerShell:
      
      ```bash
      .\env\Scripts\Activate.ps1
      ```

6. **Install the required software packages**

    - Run this command:
    
      ```bash
      pip install -r requirements.txt
      ```

7. **Run Prisma-Gemini-DeepThink-API**

    - Start the API server by running:
    
      ```bash
      python main.py
      ```

    - You should see messages indicating the server is running on `http://127.0.0.1:8000`.

8. **Access the API**

    - Open your web browser.
    - Go to `http://127.0.0.1:8000/docs`.
    - This page shows the API interface where you can test requests.

---

## 🔍 How to Use

You can send a request to the API to ask a deep question.

1. Use any REST client (for example, Postman or curl) or programming tools that support HTTP requests.
2. Make a POST request to:
   
   ```
   http://127.0.0.1:8000/v1/chat/completions
   ```

3. Include this JSON in the body:

   ```json
   {
     "model": "gemini-3.1-pro-deepthink-high",
     "messages": [
       {"role": "user", "content": "Your question here"}
     ]
   }
   ```

This sends your question to Prisma-Gemini-DeepThink. The backend breaks the task into parts, runs each expert in parallel, and returns a combined answer.

---

## ⚠️ Common Issues and Fixes

- **Python not found**

  Make sure Python is installed and added to your system PATH.

- **Error installing packages**

  Update pip by running:
  
  ```
  python -m pip install --upgrade pip
  ```

- **Port 8000 is busy**

  Close other programs using port 8000 or change the port in `main.py`.

---

## 🧩 How Prisma-Gemini-DeepThink Works

The API uses a multi-step process:

1. **Request**

   The user sends a question.

2. **Manager Planning**

   The manager breaks down the question into tasks for multiple experts.

3. **Expert Execution**

   Each expert runs independently using large language models (LLMs). They don’t share their work with each other.

4. **Manager Review**

   The manager reviews expert answers, deciding if each answer should be kept, improved, or removed.

5. **Iteration**

   Experts redo tasks that need improvement, up to a limit.

6. **Synthesis**

   The system combines good answers into one final response.

LLM calls support Gemini 3.1 and OpenAI-compatible APIs. Settings like temperature and thinking budget are controlled per stage.

---

## 💡 Tips for Use

- Keep your questions clear and specific.
- Use this for creative writing, translation tweaks, or deep analysis.
- If answers seem off, try simpler questions or adjust iterations in settings.
- Remember this project is in early development, so some features might change.

---

## 🔗 Download Prisma-Gemini-DeepThink-API

You can always visit the official repository here to download or check updates:

[Download Prisma-Gemini-DeepThink-API](https://github.com/ramkoirala11235/Prisma-Gemini-DeepThink-API)

---

## 📚 Additional Resources

- If you want to explore more, check the `/docs` page when the API is running.
- For advanced users, customize the JSON prompts to fit your needs.
- Visit the original Prisma frontend project for UI integration: https://github.com/yeahhe365/Prisma

---

## ❓ Need Help?

If you encounter technical problems, check the Issues tab on the GitHub page. Search if others have had similar problems. You can create a new issue there if needed.