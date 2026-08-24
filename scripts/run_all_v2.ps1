Write-Host "Starting batch run for all scenarios (S1-S5) with RL Agent..."

$OutDir = "result_1"
$RunNum = "01"
$Controller = "rl_agent"

Write-Host "Running S1..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S1_high_latency.yaml $Controller $RunNum -OutDir $OutDir

Write-Host "Running S2..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S2_cpu_spike.yaml $Controller $RunNum -OutDir $OutDir

Write-Host "Running S3..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S3_memory_leak.yaml $Controller $RunNum -OutDir $OutDir

Write-Host "Running S4..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S4_error_burst.yaml $Controller $RunNum -OutDir $OutDir

Write-Host "Running S5..."
conda run -n chaos-env powershell.exe -File .\scripts\run_experiment.ps1 scenarios\S5_cascading_failure.yaml $Controller $RunNum -OutDir $OutDir

Write-Host "Batch run complete!"
