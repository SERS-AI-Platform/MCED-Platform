param(
    [string]$PythonExe = "python",
    [string]$InputCsv = "results/processed_spectra.csv",
    [string]$OutputRoot = "models/results/benchmark_all_spectra",
    [string[]]$CancerTypes = @("PRO", "LUN", "CRC", "CPAN", "OVA"),
    [string[]]$NonCancerGroups = @("NOR", "DIA", "HBP", "H.D."),
    [string[]]$Models = @("resnet18", "cnn1d", "xgboost"),
    [string]$Aggregate = "none",
    [switch]$NoShap
)

$trainArgs = @(
    "models/train.py",
    "--aggregate", $Aggregate,
    "--benchmark-models"
) + $Models + @(
    "--cancer-types"
) + $CancerTypes + @(
    "--non-cancer-groups"
) + $NonCancerGroups + @(
    "-i", $InputCsv,
    "-o", $OutputRoot
)

& $PythonExe $trainArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

foreach ($model in $Models) {
    $testArgs = @(
        "models/test.py",
        "-i", (Join-Path $OutputRoot $model),
        "--processed-csv", $InputCsv
    )
    if ($NoShap) {
        $testArgs += "--no-shap"
    }

    & $PythonExe $testArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
