#!/usr/bin/env python3
"""
Test runner script for Time Series Portfolio Optimization project
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main test runner"""
    print("Time Series Portfolio Optimization - Test Suite")
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    success = True
    
    # Run unit tests
    if not run_command("python -m pytest tests/ -v --tb=short", "Unit Tests"):
        success = False
    
    # Run code formatting check
    if not run_command("python -m flake8 src/ tests/", "Code Style Check"):
        success = False
    
    # Run import sorting check
    if not run_command("python -m isort --check-only src/ tests/", "Import Sorting Check"):
        success = False
    
    if success:
        print("\n✅ All tests and checks passed!")
        return 0
    else:
        print("\n❌ Some tests or checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())