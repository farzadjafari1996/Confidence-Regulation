# Setup HuggingFace cache directory in project
# Run this after activating the virtual environment

$projectPath = Get-Location
$cachePath = Join-Path $projectPath "cache"

# Create cache directory if it doesn't exist
if (-not (Test-Path $cachePath)) {
    New-Item -ItemType Directory -Path $cachePath | Out-Null
    Write-Host "Created cache directory: $cachePath"
}

# Set HF_HOME environment variable
$env:HF_HOME = $cachePath
Write-Host "HF_HOME set to: $env:HF_HOME"
Write-Host ""
Write-Host "HuggingFace models and datasets will be cached locally in: ./cache"
