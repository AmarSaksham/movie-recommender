# 🎬 CineMatch — Movie Recommendation System

Content-based movie recommender built with **Python (pandas, scikit-learn)** and a **vanilla HTML/CSS/JS** front end. Select a movie and get the 10 most similar films using **TF-IDF + cosine similarity**. Posters load live from Wikipedia.

## Project structure
```
index.html          web app
style.css
app.js
data/movies.js      generated model output (recommendations + stats)
ml/movies_raw.csv   raw dataset (133 movies)
ml/build_model.py   cleaning → features → TF-IDF → cosine similarity → export
ml/requirements.txt
vercel.json
```

## Run locally
Open `index.html` in a browser (or use the VS Code **Live Server** extension).

## Rebuild the model (optional)
```bash
pip install -r ml/requirements.txt
python ml/build_model.py
```

## Deploy to Vercel
Push to GitHub → Vercel → **Add New Project** → import the repo → Framework Preset: **Other** → Deploy. No build command needed.
