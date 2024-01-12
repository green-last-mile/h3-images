#!/usr/bin/env python3
import subprocess
import pandas as pd
from pathlib import Path
import click


@click.command()
@click.argument("csv_file", type=click.Path(exists=True))
@click.option("--style", default="satellite-v9", help="Mapbox style for the tiles.")
@click.option(
    "--tile_res", default=512, type=int, help="Resolution of the tile (square)."
)
@click.option(
    "--output_dir",
    default="output",
    type=str,
    help="Directory to save the generated images.",
)
def main(csv_file, style, tile_res, output_dir):
    """
    Process a CSV file to generate images for each H3 index.

    The CSV file should contain columns 'label', and 'h3'.
    """
    h3_images_dir = Path.cwd()

    # Load the CSV file
    list_df = pd.read_csv(csv_file)

    # Path to the generate_image.py script
    script_path = h3_images_dir / "generate_image.py"

    # Output directory
    output_path = h3_images_dir / output_dir
    output_path.mkdir(exist_ok=True)

    # Iterate over the DataFrame and generate images
    for _, row in list_df.iterrows():
        h3_index = row["h3"]
        label = row["label"].replace(", ", "_").replace(" ", "_")

        # Define the output file path
        output_file = output_path / f"{label}.png"

        # Prepare the command
        command = [
            "python",
            str(script_path),
            "--h3",
            h3_index,
            "--output",
            str(output_file),
            "--tile-res",
            str(tile_res),
            "--style",
            style,
        ]

        # Run the command
        subprocess.run(command)

        # Check if the file was created
        if not output_file.exists():
            click.echo(f"Failed to create image for {label}(H3: {h3_index})")
        else:
            click.echo(f"Image created for {label} (H3: {h3_index})")


if __name__ == "__main__":
    main()
