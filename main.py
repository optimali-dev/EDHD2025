#!/usr/bin/env python3
"""
main.py – Application entry point
"""

import os
import sys
import logging
from dotenv import load_dotenv
from typing import Optional

# ---------------------------
# Configuration & Logging
# ---------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def load_config() -> None:
    """Load environment variables from a .env file if present."""
    load_dotenv()
    logging.info("Environment variables loaded.")

# ---------------------------
# Core Logic
# ---------------------------

def main(args: Optional[list[str]] = None) -> None:
    """
    Main entry point of the application.
    :param args: Optional list of arguments (defaults to sys.argv[1:])
    """
    if args is None:
        args = sys.argv[1:]

    logging.info("Application started with arguments: %s", args)

    try:
        # Your application logic here
        logging.info("Running main application logic...")
        print("Hello, world! 👋")
        # Example: Access an environment variable
        my_var = os.getenv("MY_VARIABLE", "default_value")
        logging.info(f"MY_VARIABLE = {my_var}")

    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        sys.exit(1)

    logging.info("Application finished successfully.")

# ---------------------------
# Script Entry
# ---------------------------

if __name__ == "__main__":
    setup_logging()
    load_config()
    main()
