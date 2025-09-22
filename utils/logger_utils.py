#!/usr/bin/env python

import psutil
import sys

def log_memory_usage(location):
    """Log current memory usage"""
    process = psutil.Process()
    rss = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
    vms = process.memory_info().vms / 1024 / 1024 / 1024  # Convert to GB
    print(f"CORDIAL Memory usage: {location}:")
    print(f"  RSS: {rss:.2f} GB, VMS: {vms:.2f} GB")
    sys.stdout.flush()  # Ensure output is written immediately
