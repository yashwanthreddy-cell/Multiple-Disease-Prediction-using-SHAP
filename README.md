# HealthAI Explainer — with Supabase Authentication

## Quick Start

### 1. Supabase setup
1. Go to https://supabase.com → open your project
2. Navigate to SQL Editor
3. Run the contents of `supabase_setup.sql` (creates the predictions table)

### 2. Credentials
Copy `.env.example` to `.env` and fill in your keys:
```
SUPABASE_URL=https://jiucaokohscxvyvkakxi.supabase.co
SUPABASE_ANON_KEY=your_actual_anon_key
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate models (if not already done)
```bash
python setup.py   # or run the 3 training notebooks
```

### 5. Run the app
```bash
streamlit run app.py
```

## File structure
```
healthai-auth/
├── app.py               ← Main app (extended with auth gate)
├── auth.py              ← Signup / Login / Logout / UI
├── db.py                ← Supabase CRUD for prediction history
├── supabase_setup.sql   ← Run once in Supabase SQL Editor
├── requirements.txt
├── .env.example         ← Copy to .env and fill credentials
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example  ← For Streamlit Cloud deployment
├── saved_models/        ← .pkl files go here
├── diabetes.csv
├── heart.csv
└── parkinsons.csv
```
