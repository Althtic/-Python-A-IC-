param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "python"
)

$marketTaskName = "QuantSystem-Tushare-EOD-Market"
$industryTaskName = "QuantSystem-Tushare-Weekly-Industry"
$financialTaskName = "QuantSystem-Tushare-Weekly-Financial"
$workdays = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

function Register-QuantTask {
    param([string]$Name, [string]$Arguments, [object]$Trigger)
    $action = New-ScheduledTaskAction -Execute $PythonExe -Argument $Arguments -WorkingDirectory $ProjectRoot
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger -Principal $principal -Settings $settings -Force | Out-Null
}

Register-QuantTask -Name $marketTaskName -Arguments "-B run_tushare_update.py market --update-factors" -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $workdays -At 18:30)
Register-QuantTask -Name $industryTaskName -Arguments "-B run_tushare_update.py market --refresh-industry" -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At 20:00)
Register-QuantTask -Name $financialTaskName -Arguments "-B run_tushare_update.py financial --financial-periods 12" -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00)

Write-Output "Registered $marketTaskName, $industryTaskName, and $financialTaskName."
