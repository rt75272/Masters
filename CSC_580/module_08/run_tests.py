import sys
import subprocess
import argparse
from pathlib import Path
"""Test runner script for the encoder-decoder model.

This script provides a convenient way to run different types of tests
with various configurations.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests  
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --fast             # Skip slow tests
"""
NUM_COLUMNS = 88

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*NUM_COLUMNS}")
    print(f"🚀 {description}")
    print(f"{'='*NUM_COLUMNS}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * NUM_COLUMNS)
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Run encoder-decoder model tests")
    parser.add_argument("--unit", action="store_true", 
                       help="Run only unit tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run only integration tests") 
    parser.add_argument("--coverage", action="store_true",
                       help="Run with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--fast", action="store_true",
                       help="Skip slow tests")
    parser.add_argument("--html-cov", action="store_true",
                       help="Generate HTML coverage report")
    args = parser.parse_args()
    base_cmd = [sys.executable, "-m", "pytest", "test_encoder_decoder.py", "test_visualize_architecture.py"]
    # Add flags based on arguments.
    if args.verbose:
        base_cmd.append("-v")
    else:
        base_cmd.append("-q")
    if args.fast:
        base_cmd.extend(["-m", "not slow"])
    # Determine test selection.
    if args.unit:
        test_classes = [
            "TestDataGenerator", 
            "TestEncoderDecoderModel", 
            "TestModelTrainer",
            "TestGPUConfig",
            "TestVisualizationDiagrams",
            "TestFileSaving",
            "TestMainFunction",
            "TestCommandLineInterface",
            "TestDiagramContent",
            "TestErrorHandling"]
        for test_class in test_classes:
            cmd = base_cmd + [f"::{test_class}"]
            run_command(cmd, f"Running {test_class} tests")
    elif args.integration:
        integration_classes = ["TestIntegration"]
        for test_class in integration_classes:
            cmd = base_cmd + [f"::{test_class}"]
            run_command(cmd, f"Running {test_class} tests")
    else:
        # Run all tests.
        cmd = base_cmd.copy()
        # Add coverage if requested.
        if args.coverage or args.html_cov:
            cmd.extend(["--cov=.", "--cov-report=term-missing"])
            if args.html_cov:
                cmd.extend(["--cov-report=html"])
        result = run_command(cmd, "Running all tests")
        if args.html_cov and result == 0:
            print(f"\n📊 HTML coverage report generated in 'htmlcov/index.html'")
    print(f"\n{'='*NUM_COLUMNS}")
    print("🎉 Testing completed.")
    print(f"{'='*NUM_COLUMNS}")

# The big red activation button.
if __name__ == "__main__":
    main()
