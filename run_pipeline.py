"""One command to rebuild everything: data -> cleaning -> analysis -> SQL -> PDF report.

    python run_pipeline.py
"""
import subprocess
import sys

STEPS = ["src/generate_data.py", "src/clean.py", "src/analysis.py", "src/run_sql.py", "reports/build_report.py"]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, step], check=True)
print("\nDone. Launch the dashboard with:  streamlit run app.py")
