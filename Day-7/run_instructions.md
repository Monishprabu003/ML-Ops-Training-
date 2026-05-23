# Placement Mentor AI - Run Instructions

Welcome to your new ML project! Follow these simple steps to run your Flask application in VS Code.

## Prerequisites
Make sure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/).

## Steps to Run

1. **Open the Project in VS Code**
   Open the folder `Placement_Mentor_AI` in Visual Studio Code.

2. **Open the Terminal**
   In VS Code, open a new terminal by clicking on `Terminal > New Terminal` from the top menu, or by pressing `Ctrl + \``.

3. **Create a Virtual Environment (Recommended)**
   It's good practice to create a virtual environment to manage dependencies. Run:
   ```bash
   python3 -m venv venv
   ```

4. **Activate the Virtual Environment**
   - **On Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```cmd
     .\venv\Scripts\activate
     ```

5. **Install Dependencies**
   Install all the required Python packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Flask App**
   Start the application by running the `app.py` script:
   ```bash
   python app.py
   ```

7. **View the Application**
   Once the server starts, you will see output like `Running on http://127.0.0.1:5000`. 
   Cmd+Click (Mac) or Ctrl+Click (Windows) the link `http://127.0.0.1:5000` in the terminal to open the web application in your browser.

## Troubleshooting
- **Error loading model**: Ensure `model.pkl` is located in the same directory as `app.py`.
- **Port already in use**: If port 5000 is taken, you can change it at the bottom of `app.py` in `app.run(debug=True, port=5001)`.
