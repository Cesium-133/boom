import json
import os
import argparse
import sys
import plotly.io as pio


def _import_visualizer():
    # Support running as a package or from workspace root
    try:
        from .visualization import SimulationVisualizer  # type: ignore
        return SimulationVisualizer
    except Exception:
        from visualization import SimulationVisualizer  # type: ignore
        return SimulationVisualizer


def main():
    """Main function to run the simulation."""
    # Ensure we render via the default web browser
    try:
        pio.renderers.default = "browser"
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Missile Simulation Runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration JSON file (e.g., missile_simulation/config.json)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        # Resolve relative to CWD first, then relative to this file if not found
        cwd_path = os.path.join(os.getcwd(), config_path)
        module_path = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(cwd_path):
            config_path = cwd_path
        elif os.path.exists(module_path):
            config_path = module_path

    if not os.path.exists(config_path):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    SimulationVisualizer = _import_visualizer()
    visualizer = SimulationVisualizer(config)
    fig = visualizer.create_animation()

    print("Showing simulation... Close the browser tab to exit.")
    fig.show()

if __name__ == "__main__":
    main()


