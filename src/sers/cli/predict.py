"""sers predict — Run inference on spectrum files."""

import click

from ._run import run_script


@click.command()
@click.argument("spectra", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--age", type=float, default=None, help="Patient age.")
@click.option("--sex", default=None, help="Patient sex (M/F).")
@click.option("--bmi", type=float, default=None, help="Patient BMI.")
@click.option("--mode", "-m",
              type=click.Choice(["screening", "balanced", "confirmatory"]),
              default=None, help="Operating mode.")
@click.option("--instrument", "-i",
              type=click.Choice(["thermo", "medical"]),
              default=None, help="Source instrument [thermo].")
@click.option("--model-dir", default=None, help="Model artifacts directory.")
@click.option("--output", "-o", default=None, help="Output JSON path.")
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Suppress text, JSON only.")
def predict(spectra, age, sex, bmi, mode, instrument, model_dir, output, quiet):
    """Run cancer screening inference on spectrum CSV file(s).

    \b
    Examples:
        sers predict spectrum.csv
        sers predict *.CSV --age 55 --sex M --bmi 24.3
        sers predict sample.csv --mode screening --instrument medical
        sers predict batch/*.CSV -o results.json -q
    """
    args = list(spectra)
    simple = {
        "--age": age,
        "--sex": sex,
        "--bmi": bmi,
        "--mode": mode,
        "--instrument": instrument,
        "--model-dir": model_dir,
        "--output": output,
    }
    for flag, val in simple.items():
        if val is not None:
            args += [flag, str(val)]
    if quiet:
        args.append("--quiet")
    run_script("scripts/deployment/sers_predict.py", args)
