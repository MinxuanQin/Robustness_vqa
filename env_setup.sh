py_venv_dir="${SCRATCH}/.python_venv"
python -m venv ${py_venv_dir}/vqa --upgrade-deps
${SCRATCH}/.python_venv/vqa/bin/pip install -r requirements.txt --cache-dir ${SCRATCH}/pip_cache