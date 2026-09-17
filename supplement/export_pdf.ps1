# Update the table of contents / page numbers of results/supplement/webappendix.docx
# in Word and export it as PDF (Lancet: one PDF, table of contents, numbered pages).
#   powershell -ExecutionPolicy Bypass -File supplement/export_pdf.ps1
param(
  [string]$Docx = (Join-Path $PSScriptRoot "..\results\supplement\webappendix.docx"),
  [string]$Pdf  = (Join-Path $PSScriptRoot "..\results\supplement\webappendix.pdf")
)
$Docx = (Resolve-Path $Docx).Path
$Pdf  = [System.IO.Path]::GetFullPath($Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($Docx, $false, $false)
  $doc.Fields.Update() | Out-Null
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() | Out-Null }
  $doc.Repaginate()
  $pages = $doc.ComputeStatistics(2)          # wdStatisticPages
  $doc.Save()
  $doc.ExportAsFixedFormat($Pdf, 17, $false, 0, 0, 0, 0, 0, $true, $true, 1, $true, $true, $false)
  $doc.Close($false)
  "exported $Pdf ($pages pages)"
} finally {
  $word.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
