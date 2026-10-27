param([string]$Project='')
$ErrorActionPreference='Stop'
try {
  if(-not $Project -and (Test-Path -LiteralPath "$PSScriptRoot\.local\desktop_settings.json")) {
    $settings=Get-Content -LiteralPath "$PSScriptRoot\.local\desktop_settings.json" -Raw -Encoding utf8 | ConvertFrom-Json
    $Project=$settings.project
  }
  if(-not $Project) {$Project=$env:GPT_SOVITS_HOME}
  if(-not $Project -or -not (Test-Path -LiteralPath "$Project\runtime\pythonw.exe")) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog=New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description='Select the existing GPT-SoVITS installation folder'
    if($dialog.ShowDialog() -ne 'OK'){exit 0}
    $Project=$dialog.SelectedPath
  }
  if(-not (Test-Path -LiteralPath "$Project\runtime\pythonw.exe")) {throw 'runtime/pythonw.exe was not found in the selected installation.'}
  $env:PYTHONIOENCODING='utf-8'
  $arguments=@('-B',"`"$PSScriptRoot\tools\desktop.py`"",'--project',"`"$Project`"")
  Start-Process -FilePath "$Project\runtime\pythonw.exe" -ArgumentList $arguments -WorkingDirectory $Project -WindowStyle Hidden
} catch {
  Add-Type -AssemblyName System.Windows.Forms
  [System.Windows.Forms.MessageBox]::Show($_.Exception.Message,'Voice Generate') | Out-Null
  exit 1
}
