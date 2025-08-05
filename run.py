import sys
import importlib

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <command> [options]")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    try:
        # Try to import the script dynamically
        module = importlib.import_module(f"scripts.{command}")
    except ModuleNotFoundError as e:
        print(e)
        print(f"Unknown command: {command}")
        sys.exit(1)

    sys.argv = [f"run.py {command}"] + args  # Fix argv for argparse inside the script
    module.main()

if __name__ == "__main__":
    main()
