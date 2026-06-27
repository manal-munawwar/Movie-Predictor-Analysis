# 🎬 CINECAST — Movie Success Predictor
### Great Lakes Institute of Management

> An AI-powered movie investment intelligence tool that predicts whether a film
> will be commercially successful (Revenue > Budget) using a Gradient Boosting
> classifier trained on TMDB + IMDb data.
>
> **Model Performance:** Test AUC = 0.874 | F1 Score = 0.861 | Accuracy = 86.9%

---

## 📁 Project Structure

After setup, your `Movie-Predictor-Analysis` folder should contain these files:

```
Movie-Predictor-Analysis/
│
├── app.py                  ← Flask backend (REST API server)
├── streamlit_app.py        ← Streamlit frontend (the visual app)
├── requirements.txt        ← All Python dependencies
├── model.pkl               ← Trained Gradient Boosting model
├── scaler.pkl              ← Fitted StandardScaler
├── feature_names.pkl       ← Feature column order list
└── README.md               ← This file
```

> **Note:** `model.pkl`, `scaler.pkl`, and `feature_names.pkl` are generated
> by running the export cell in your Jupyter notebook. See Step 2 below.

---

## 🛠️ Setup Guide (Step by Step)

Follow these steps in order. Every command is typed into your **Terminal**
(or the VS Code terminal).

---

### STEP 1 — Make Sure Python Is Installed

Open your Terminal and check your Python version:

```bash
python3 --version
```

You should see something like `Python 3.14.x`. If you see an error, download
Python from https://www.python.org/downloads/

---

### STEP 2 — Install All Dependencies

Navigate into your project folder:

```bash
cd /Users/YOUR_USERNAME/Desktop/CAPSTONE/Movie-Predictor-Analysis
```

> Replace `YOUR_USERNAME` with your actual Mac username.
> For example: `cd /Users/manalmunawwar/Desktop/CAPSTONE/Movie-Predictor-Analysis`

Then install all required packages in one command:

```bash
pip3 install -r requirements.txt
```

Wait for all packages to finish installing. You will see a lot of text scroll
by — this is normal. It is done when you see your terminal prompt again.

**Verify everything installed correctly:**

```bash
python3 -c "import sklearn, pandas, numpy, joblib, flask, streamlit, plotly, requests; print('All packages OK!')"
```

You should see: `All packages OK!`

---

### STEP 3 — Generate the Model Files (One Time Only)

> **Skip this step if you already have `model.pkl`, `scaler.pkl`, and
> `feature_names.pkl` in your project folder.**

The app needs three `.pkl` files that are exported from your Jupyter notebook.
You only need to do this once.

**3a.** Open your notebook in VS Code:
```
Movie_Success_Prediction_Interim.ipynb
```

**3b.** Make sure the kernel (top right of VS Code) is set to:
```
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
```

**3c.** Run all cells from top to bottom using:
- `Run All` button at the top, OR
- `Shift + Enter` on each cell one by one

**3d.** Add a new code cell at the very bottom of the notebook and paste
the contents of `export_model_cell.py` into it. Run that cell.

**3e.** You should see:
```
✅ model.pkl saved
✅ scaler.pkl saved
✅ feature_names.pkl saved
🎬 ALL FILES EXPORTED SUCCESSFULLY
```

**3f.** Move the three generated files into your project folder:
```bash
mv ~/Desktop/CAPSTONE/model.pkl ~/Desktop/CAPSTONE/Movie-Predictor-Analysis/
mv ~/Desktop/CAPSTONE/scaler.pkl ~/Desktop/CAPSTONE/Movie-Predictor-Analysis/
mv ~/Desktop/CAPSTONE/feature_names.pkl ~/Desktop/CAPSTONE/Movie-Predictor-Analysis/
```

**Verify the files are there:**
```bash
ls ~/Desktop/CAPSTONE/Movie-Predictor-Analysis/*.pkl
```

You should see three files listed: `feature_names.pkl`, `model.pkl`, `scaler.pkl`

---

### STEP 4 — Start the Flask Backend (Terminal 1)

The Flask backend is the engine that runs the prediction model.
It must be running before you launch the Streamlit app.

Open a terminal and run:

```bash
cd /Users/manalmunawwar/Desktop/CAPSTONE/Movie-Predictor-Analysis
python3 app.py
```

**You should see:**
```
🎬 Movie Success Predictor — Flask API
✅ Model loaded — 200 estimators, 38 features
✅ Scaler loaded
✅ 38 feature names loaded
🚀 Starting Flask server on http://127.0.0.1:5000
```

> ⚠️ **Keep this terminal running.** Do NOT close it while using the app.
> The Flask server must stay active in the background.

**Test the server is working** (optional but recommended):

Open a second terminal and run:
```bash
curl http://127.0.0.1:5000/health
```

You should see a JSON response showing `"status": "running"`.

---

### STEP 5 — Launch the Streamlit App (Terminal 2)

Open a **new terminal window** (keep Terminal 1 running) and type:

```bash
cd /Users/manalmunawwar/Desktop/CAPSTONE/Movie-Predictor-Analysis
streamlit run streamlit_app.py
```

**You should see:**
```
You can now view your Streamlit app in your browser.
Local URL:  http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Your browser will open automatically at `http://localhost:8501`.

If it does not open automatically, copy `http://localhost:8501` and
paste it into your browser manually.

---

## 🎬 How to Use the App

Once the app is open in your browser:

### Making a Prediction

1. **Fill in the sidebar** on the left with your film's parameters:
   - Production Budget (in USD Millions)
   - TMDB Popularity Score
   - Expected IMDb Rating
   - IMDb Vote Count
   - Runtime, Release Month, Release Year
   - Genres (you can select multiple)
   - Language, Adult Content, Tagline flags
   - Cast size, Directors, Writers, Production Companies

2. **Click the red button** at the bottom of the sidebar:
   ```
   🎬 PREDICT SUCCESS
   ```

3. **Read your verdict:**
   - ✅ **GREENLIGHT** — model predicts commercial success (Revenue > Budget)
   - ❌ **PASS** — model predicts the film will not recoup its budget

4. **Check the confidence score** — the colour-coded bar tells you how
   certain the model is:
   - 🟢 Green (> 70%) — High confidence
   - 🟡 Amber (50–70%) — Borderline prediction
   - 🔴 Red (< 50%) — Low confidence

5. **Read the explanation** — the plain-English section below the chart
   tells you exactly why the model made this prediction.

### Running the Example Scenarios

Click **Run Example** under any of the four pre-built scenarios:
- 🦁 **Blockbuster** — $250M action/adventure
- 🎭 **Indie Drama** — $2M critical drama
- 👻 **Horror Hit** — $15M horror/thriller
- 🌍 **Foreign Film** — $5M non-English arthouse

These are great for demonstrating the app to mentors and evaluators.

---

## 🔁 Stopping and Restarting the App

### To stop the app:
- In **Terminal 1** (Flask): Press `Ctrl + C`
- In **Terminal 2** (Streamlit): Press `Ctrl + C`

### To restart the app:
- **Terminal 1:** `python3 app.py`
- **Terminal 2:** `streamlit run streamlit_app.py`

---

## 🌐 Deploying to Streamlit Community Cloud (Optional)

To make the app publicly accessible (e.g. for sharing with your mentor):

> **Note:** Streamlit Community Cloud only hosts the Streamlit frontend.
> For a full cloud deployment including the Flask backend, you would need
> a service like Railway, Render, or Heroku for the API.
> For academic demos, running both locally is perfectly sufficient.

**Steps for Streamlit Cloud (frontend only):**

1. Create a free account at https://streamlit.io/cloud

2. Push your project to a GitHub repository:
   ```bash
   git init
   git add streamlit_app.py requirements.txt
   git commit -m "Initial commit — Cinecast Movie Predictor"
   git remote add origin https://github.com/YOUR_USERNAME/cinecast.git
   git push -u origin main
   ```

3. In Streamlit Cloud, click **New App** → connect your GitHub repo →
   set the main file to `streamlit_app.py` → Deploy

4. Update the `API_URL` variable in `streamlit_app.py` to point to your
   deployed Flask backend URL instead of `http://127.0.0.1:5000/predict`

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Library | scikit-learn GradientBoostingClassifier |
| Training Data | TMDB + IMDb dataset (~4,800 films) |
| Target Variable | Revenue > Budget (binary: 1 = Successful) |
| Number of Features | 38 engineered features |
| n_estimators | 200 |
| max_depth | 4 |
| learning_rate | 0.05 |
| Test AUC | 0.874 |
| F1 Score | 0.861 |
| Accuracy | 86.9% |
| Class Split | ~67% Successful / ~33% Unsuccessful |

### Top 5 Most Important Features

| Rank | Feature | Importance |
|---|---|---|
| 1 | Log_Num_Votes (Audience Engagement) | 0.1641 |
| 2 | Log_Budget (Production Budget) | 0.1282 |
| 3 | Movie_Age (Film Age / Legacy) | 0.1006 |
| 4 | Num_Votes (Raw Vote Count) | 0.0967 |
| 5 | Release_Year | 0.0900 |

---

## ❓ Troubleshooting

**"Module not found" error:**
```bash
pip3 install -r requirements.txt
```

**"Cannot connect to Flask server" in the browser:**
- Make sure `python3 app.py` is running in a separate terminal
- Check that nothing else is using port 5000:
  ```bash
  lsof -i :5000
  ```
- If another process is using port 5000, stop it or change the port in
  `app.py` (line: `app.run(debug=True, port=5000)`) and update `API_URL`
  in `streamlit_app.py` to match

**Streamlit app not opening in browser:**
- Manually go to `http://localhost:8501` in your browser

**VS Code not finding the right Python:**
- Press `Cmd + Shift + P` → type `Python: Select Interpreter`
- Choose `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`

**Port 8501 already in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## 📚 References

Key papers and sources cited in the project report:
- Asur & Huberman (2010) — Predicting the Future with Social Media
- Quader et al. (2017) — A Machine Learning Approach to Predict Movie Box-Office Success
- Breiman (2001) — Random Forests
- Chen & Guestrin (2016) — XGBoost
- Chawla et al. (2002) — SMOTE
- He & Garcia (2009) — Learning from Imbalanced Data

---

*Built by Manal Munawwar
