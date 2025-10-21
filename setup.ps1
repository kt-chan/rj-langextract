# Check if Conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda is not installed or not available in PATH."
    exit 1
}

# Check if requirements.txt exists
if (-not (Test-Path requirements.txt)) {
    Write-Error "requirements.txt not found in current directory."
    exit 1
}

# Create or check Conda environment
$venvPath = ".\venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment exists at $venvPath. Checking if Python version is up to date..."
    conda update --prefix $venvPath --yes python=3.10
} else {
    Write-Host "Creating conda environment at $venvPath..."
    conda create -p $venvPath  --yes python=3.10 
}

# Get the Python executable path
$pythonPath = Join-Path $venvPath "python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = Join-Path $venvPath "bin/python"
}

# Install or upgrade all packages using uv with the specified Python
Write-Host "Installing or upgrading packages with conda..."
conda activate $venvPath
pip install --upgrade -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/simple/ -i https://mirrors.ustc.edu.cn/anaconda/pkgs/free/ -i https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/ --trusted-host mirrors.ustc.edu.cn

Write-Host "Setup completed successfully."