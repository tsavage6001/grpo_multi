# to start the program

python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

pip install torch transformers datasets accelerate peft trl pandas openai


####################
brew install xz
env PYTHON_CONFIGURE_OPTS="--with-lzma" pyenv install 3.10.13 --force
pyenv global 3.10.13


# setup.sh

# Use correct Python version (already compiled with lzma support)
PYTHON_BIN="$(pyenv which python3.10)"

# Create a clean virtual environment
$PYTHON_BIN -m venv .venv
source .venv/bin/activate

# Upgrade pip and core packaging tools
pip install --upgrade pip setuptools wheel

# Install Python libraries
pip install torch transformers datasets accelerate peft trl pandas openai


PYTHONPATH=/Users/thomassavage/grpo_trs2 python training_run_trs.py
