$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $repositoryRoot 'config\thesis-sources.json'
$manifestDirectory = Split-Path -Parent $manifestPath
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json

foreach ($source in $manifest.sources) {
    if (-not $source.url -or -not $source.local_path) {
        throw "Each source needs a URL and local_path."
    }
    $destination = [IO.Path]::GetFullPath((Join-Path $manifestDirectory $source.local_path))
    if (-not $destination.StartsWith((Join-Path $repositoryRoot 'data') + [IO.Path]::DirectorySeparatorChar)) {
        throw "Source destination must stay inside the repository data directory."
    }
    if (Test-Path -LiteralPath $destination) {
        Write-Host "Already present: $destination"
        continue
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $source.url -OutFile $destination
    Write-Host "Downloaded: $destination"
}
