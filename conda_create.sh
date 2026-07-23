
set -e  
DEFAULT_ENV_NAME=".conda"

usage() {
    echo "Usage: $0 [ENV_NAME]"
    echo "  ENV_NAME  (optional) Name of the Conda environment. Default: $DEFAULT_ENV_NAME"
    exit 1
}

if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH."
    echo "Please install Conda first (e.g., via Miniconda or Anaconda)."
    exit 1
fi

ENV_NAME="${1:-$DEFAULT_ENV_NAME}"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Error: Environment '$ENV_NAME' already exists."
    echo "To remove it, run: conda env remove -n $ENV_NAME"
    exit 1
fi

echo "Creating Conda environment '$ENV_NAME' with Python 3.11..."
conda create -n "$ENV_NAME" python=3.11 -y

echo "Environment '$ENV_NAME' created successfully."
echo "To activate it, run: conda activate $ENV_NAME"