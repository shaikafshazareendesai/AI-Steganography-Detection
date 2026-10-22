# Quick Start

1. Install Python 3.10 or 3.11.
2. Create and activate a virtual environment.
3. Install the core packages:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
streamlit run app.py
```

5. Visit `http://localhost:8501`.
6. Use **Hide a message** to create a PNG, then upload that PNG in **Analyze image** to verify extraction.

## Optional CNN experiment

```bash
pip install -r requirements-ml.txt
python generate_dataset.py
python train.py
```

The generated images are synthetic. Treat this as an educational baseline, not a validated detector. No trained model is included by default.
