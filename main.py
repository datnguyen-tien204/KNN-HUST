import subprocess
import importlib


def install_requirements(requirements_file):
    with open(requirements_file, 'r') as file:
        requirements = file.read().splitlines()

    for requirement in requirements:
        try:
            importlib.import_module(requirement)
            print(f"{requirement} installed.")
        except ImportError:
            print(f"{requirement} not install. Waiting for a minutes")
            subprocess.call(["pip", "install", requirement])
            print(f"{requirement} install successfully.")
requirements_file = "requirements.txt"
install_requirements(requirements_file)


def run_streamlit_app(app_file):
    cmd = f"streamlit run {app_file}"
    subprocess.call(cmd, shell=True)

app_file = "app.py"
run_streamlit_app(app_file)
