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
    conda install --prefix $venvPath --yes python=3.10
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
pip install --upgrade -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple -i https://mirrors.ustc.edu.cn/pypi/simple/ -i https://mirrors.ustc.edu.cn/anaconda/pkgs/free/ -i https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/ --trusted-host mirrors.ustc.edu.cn

# Write-Host "Installing or upgrading langextract-glmprovider ..."

# python -m pip install --upgrade pip setuptools wheel
# pip install --upgrade --no-cache-dir -e langextract-glmprovider
$env:SSL_CERT_FILE="D:\certs\ca-bundle.crt"
$env:REQUESTS_CA_BUNDLE="D:\certs\ca-bundle.crt"
$env:SETUPTOOLS_USE_DISTUTILS = "stdlib"
pip install --upgrade --no-cache-dir --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple -e langextract-glmprovider

Write-Host "Setup completed successfully."