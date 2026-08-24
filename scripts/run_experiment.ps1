param (
    [Parameter(Mandatory=$true)][string]$ScenarioFile,
    [Parameter(Mandatory=$true)][string]$Controller,
    [Parameter(Mandatory=$true)][string]$RunNum,
    [Parameter(Mandatory=$false)][string]$OutDir = "results/raw"
)

$ScenarioName = (Get-Item $ScenarioFile).BaseName
$RunID = "${ScenarioName}-${Controller}-${RunNum}"

Write-Host "=========================================="
Write-Host "Starting Experiment Run: $RunID"
Write-Host "=========================================="

# 1. Reset chaos (clean state)
Write-Host "Resetting chaos state..."
python scripts/chaos_reset.py

# 2. Record start time in UTC format compatible with Prometheus (ISO 8601 with Z)
$StartTime = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Write-Host "Start time recorded: $StartTime"

# 3. Warmup (3 mins to establish baseline metrics)
Write-Host "Warming up for 3 minutes..."
Start-Sleep -Seconds 180

# 4. Inject fault according to scenario
Write-Host "Starting fault injection timeline..."
python scripts/inject_fault.py --scenario $ScenarioFile --run-id $RunID --out-dir $OutDir

# 5. Record end time (give extra 2 mins for metrics to settle after last step)
Write-Host "Cooldown period (2 minutes)..."
Start-Sleep -Seconds 120
$EndTime = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

# Reset chaos again at the very end
python scripts/chaos_reset.py

# 6. Export metrics
Write-Host "Exporting metrics to CSV..."
python scripts/export_data.py --run-id $RunID --start $StartTime --end $EndTime --out-dir $OutDir

Write-Host "=========================================="
Write-Host "Run $RunID Complete!"
Write-Host "Data saved in $OutDir/$RunID/"
Write-Host "=========================================="
