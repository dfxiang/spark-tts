param(
    [string]$ModelDir = "D:\models\spark-tts",
    [string]$ArtifactDir = (Join-Path $PSScriptRoot "artifacts"),
    [string]$Image = "spark-tts-trtllm-api:bf16",
    [string]$BaseImage = "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3",
    [string]$ContainerName = "spark-tts-trtllm-api",
    [int]$HttpPort = 8168,
    [int]$GrpcPort = 8169,
    [int]$MetricsPort = 8170
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ModelDir = (Resolve-Path $ModelDir).Path
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

docker build `
    --build-arg "BASE_IMAGE=$BaseImage" `
    -f (Join-Path $ProjectRoot "Dockerfile.cuda") `
    -t $Image `
    $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build the TensorRT-LLM sidecar image."
}

$Legacy = docker ps --filter "name=^/spark-tts$" --format "{{.ID}}"
if ($Legacy) {
    docker stop spark-tts | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to stop the existing spark-tts container."
    }
}

$Existing = docker ps -a --filter "name=^/$ContainerName$" --format "{{.ID}}"
if ($Existing) {
    docker rm -f $ContainerName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to replace existing container $ContainerName."
    }
}

docker run -d `
    --name $ContainerName `
    --gpus "device=0" `
    --shm-size=4g `
    -p "${HttpPort}:8000" `
    -p "${GrpcPort}:8101" `
    -p "${MetricsPort}:8102" `
    -v "${ModelDir}:/models/Spark-TTS-0.5B:ro" `
    -v "spark-tts-voices:/data/voices" `
    -v "${ArtifactDir}:/artifacts" `
    $Image

if ($LASTEXITCODE -ne 0) {
    throw "Failed to start the TensorRT-LLM sidecar container."
}

Write-Host "Started $ContainerName on http://127.0.0.1:${HttpPort}"
Write-Host "Follow startup logs with: docker logs -f $ContainerName"
