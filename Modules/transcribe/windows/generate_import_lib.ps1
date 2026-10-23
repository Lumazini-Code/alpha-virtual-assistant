# Gera uma biblioteca de importação (.lib) a partir de um .dll que não vem com .lib.
#
# Uso (a partir do "x64 Native Tools Command Prompt for VS", onde dumpbin/lib estão no PATH):
#   powershell -ExecutionPolicy Bypass -File .\generate_import_lib.ps1 -DllName transcribe.dll
#
# O script:
#   1. Extrai os nomes exportados do DLL com dumpbin /exports
#   2. Cria um .def temporário com esses nomes
#   3. Gera o <nome>.lib com lib /def:...

param(
    [Parameter(Mandatory = $true)]
    [string]$DllName
)

$ErrorActionPreference = "Stop"

# Ferramentas do Visual Studio devem estar no PATH (rodar pelo Native Tools Command Prompt)
foreach ($tool in @("dumpbin.exe", "lib.exe")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "$tool nao encontrado. Abra o 'x64 Native Tools Command Prompt for VS' antes de rodar este script."
    }
}

$DllPath = Join-Path $PSScriptRoot $DllName
if (-not (Test-Path $DllPath)) {
    Write-Error "DLL nao encontrado: $DllPath"
}

Write-Host "Extraindo exportacoes de $DllName ..."

# Linhas de 'dumpbin /exports' tem o formato:
#     ordinal    hint    RVA        name
#           1    0      00001D60   nomeDaFuncao
$exportLines = & dumpbin /exports $DllPath |
    Select-String -Pattern '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$'

if ($exportLines.Count -eq 0) {
    Write-Error "Nenhuma exportacao encontrada em $DllName (exports por ordinal?)."
}

$names = $exportLines | ForEach-Object { $_.Matches[0].Groups[1].Value }
Write-Host ("Encontradas {0} exportacoes." -f $names.Count)

$libBaseName = [System.IO.Path]::GetFileNameWithoutExtension($DllName)
$defPath = Join-Path $PSScriptRoot ($libBaseName + ".def")

# Monta o .def (mantem os nomes exatamente como exportados, inclusive nomes manipulados/mangled)
$def = @("LIBRARY $libBaseName", "EXPORTS") + $names
$def | Set-Content -Path $defPath -Encoding ASCII

$libPath = Join-Path $PSScriptRoot ($libBaseName + ".lib")

Write-Host "Gerando $libBaseName.lib a partir de $defPath ..."

& lib.exe "/def:$defPath" /machine:x64 "/out:$libPath"

if ($LASTEXITCODE -ne 0) {
    Write-Error "lib.exe falhou com o codigo $LASTEXITCODE."
}

Write-Host "OK: $libBaseName.lib gerado em $PSScriptRoot"
