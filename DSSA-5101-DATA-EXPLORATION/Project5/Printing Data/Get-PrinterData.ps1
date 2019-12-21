## Downloads a copy of printer data from PCounter and PaperCut servers, and compresses them into an archive
## Version 1.00

$pcounter_printers = @("acprint2","acprint4","acprint6","acprint8")
$papercut_printers = @("fsprint2","fsprint3")

$dest = New-Item -Type Directory -Name "Printer Data RAW $(Get-Date -Format MM-dd-yy)" -Force

#PaperCut PrintLogger
try {
    Foreach($printer in $papercut_printers) {
        Get-ChildItem -Filter "*.csv" -Path "\\$printer.stockton.edu\c$\Program Files (x86)\PaperCut Print Logger\logs\csv\monthly" | foreach {
            Copy-Item -Path $_.FullName -Destination "$dest/$($printer)_$($_.Name)"
        }
        Write-Host "Finished $printer"
    }
} Catch {
    Write-Error "Error processing $printer"
}

#PCounter
Try {
    Foreach($printer in $pcounter_printers) {
        Get-ChildItem -Filter "PCOUNTER*.LOG" -Path "\\$printer.ac.stockton\pcounterdata$" | foreach {
            Copy-Item -Path $_.FullName -Destination "$dest/$($printer)_$($_.Name)"
        }
        Write-Host "Finished $printer"
    }
} Catch {
    Write-Error "Error processing $printer"
}

Try {
    Compress-Archive -Path $dest -DestinationPath "$($dest.Name).zip" -CompressionLevel Optimal
    Write-Host "Finished compressing archive"
} Catch {
    Write-Error "Error compressing archive"
}
