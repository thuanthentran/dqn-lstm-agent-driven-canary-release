Write-Host "Starting batch run for S2, S3, S4..."

Write-Host "Running S2..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S2_cpu_spike.yaml rl_agent 01

Write-Host "Running S3..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S3_memory_leak.yaml rl_agent 01

Write-Host "Running S4..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S4_error_burst.yaml rl_agent 01

Write-Host "Batch run complete!"
