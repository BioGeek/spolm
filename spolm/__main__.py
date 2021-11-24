"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """spolm."""


if __name__ == "__main__":
    main(prog_name="spolm")  # pragma: no cover
