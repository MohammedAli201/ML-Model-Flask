"""Backward-compatible training entry point."""
from train import train

if __name__ == "__main__":
    _, accuracy, count = train()
    print(f"Held-out accuracy: {accuracy:.3f} on {count} rows")
