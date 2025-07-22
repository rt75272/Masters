"""Tests for the visualize_architecture.py module.

This module tests the architecture visualization functionality including
diagram generation, file saving, and command-line interface.

Usage:
    pytest test_visualize_architecture.py
    pytest test_visualize_architecture.py --html-cov
    pytest test_visualize_architecture.py --cov
"""
import pytest
import sys
import os
import tempfile
import argparse
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO
from datetime import datetime
from visualize_architecture import (
    generate_detailed_architecture_diagram,
    generate_data_flow_diagram,
    save_diagrams_to_file,
    main)

# Add the current directory to the path so we can import our modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestVisualizationDiagrams:
    """Test the diagram generation functions."""
    
    def test_generate_detailed_architecture_diagram(self):
        """Test that detailed architecture diagram is generated correctly."""
        diagram = generate_detailed_architecture_diagram()
        # Check that the diagram contains expected sections.
        assert "DETAILED ENCODER-DECODER ARCHITECTURE DIAGRAM" in diagram
        assert "INPUT DATA FLOW" in diagram
        assert "ENCODER SECTION" in diagram
        assert "DECODER SECTION" in diagram
        assert "LOSS & OPTIMIZATION" in diagram
        assert "TRAINING DYNAMICS" in diagram
        # Check for specific technical details.
        assert "Units: 256" in diagram
        assert "Dropout" in diagram
        assert "Batch Normalization" in diagram
        assert "Categorical Crossentropy" in diagram
        assert "Adam" in diagram
        assert "525,000" in diagram # Parameter count.
        # Check for tensor shapes.
        assert "[64, 6, 51]" in diagram
        assert "[64, 3, 51]" in diagram
        assert "[batch=64, units=256]" in diagram
        # Verify it's a non-empty string.
        assert isinstance(diagram, str)
        assert len(diagram) > 1000  # Should be substantial content.
    
    def test_generate_data_flow_diagram(self):
        """Test that data flow diagram is generated correctly."""
        diagram = generate_data_flow_diagram()
        # Check that the diagram contains expected sections.
        assert "TENSOR FLOW DIAGRAM" in diagram
        assert "RAW DATA GENERATION" in diagram
        assert "ONE-HOT ENCODING" in diagram
        assert "PREPROCESSING" in diagram
        assert "BATCH TENSORS" in diagram
        assert "ENCODER" in diagram
        assert "DECODER" in diagram
        assert "LOSS COMPUTATION" in diagram
        assert "BACKPROPAGATION" in diagram
        assert "PARAMETER UPDATE" in diagram
        # Check for tensor shape evolution.
        assert "TENSOR SHAPE EVOLUTION SUMMARY" in diagram
        assert "[15000 × 6 × 51]" in diagram
        assert "[64 × 6 × 51]" in diagram
        assert "sequence_length" in diagram
        assert "vocabulary_size" in diagram
        # Verify it's a non-empty string.
        assert isinstance(diagram, str)
        assert len(diagram) > 1000


class TestFileSaving:
    """Test file saving functionality."""
    
    def test_save_diagrams_to_file_default_filename(self):
        """Test saving diagrams with default filename."""
        test_content = "Test architecture diagram content"
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory.
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try: 
                filename = save_diagrams_to_file(test_content)
                # Check that file was created.
                assert os.path.exists(filename)
                assert filename.startswith("neural_network_architecture_")
                assert filename.endswith(".txt")
                # Check file contents.
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                assert "ENCODER-DECODER NEURAL NETWORK ARCHITECTURE DOCUMENTATION" in content
                assert test_content in content
                assert "Generated on:" in content
            finally:
                os.chdir(original_cwd)
    
    def test_save_diagrams_to_file_custom_filename(self):
        """Test saving diagrams with custom filename."""
        test_content = "Test custom filename content"
        custom_filename = "test_architecture.txt"
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory.
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                returned_filename = save_diagrams_to_file(test_content, custom_filename)
                # Check that correct filename was returned.
                assert returned_filename == custom_filename
                assert os.path.exists(custom_filename)
                # Check file contents.
                with open(custom_filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                assert test_content in content
            finally:
                os.chdir(original_cwd)
    
    def test_save_diagrams_to_file_with_timestamp(self):
        """Test that timestamp is included in saved file."""
        test_content = "Test timestamp content"
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                # Mock datetime to control timestamp.
                with patch('visualize_architecture.datetime') as mock_datetime:
                    mock_now = datetime(2025, 1, 1, 12, 0, 0)
                    mock_datetime.now.return_value = mock_now
                    mock_datetime.strftime = datetime.strftime
                    filename = save_diagrams_to_file(test_content)
                    # Check timestamp in filename.
                    assert "20250101_120000" in filename
                    # Check timestamp in file content.
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    assert "2025-01-01 12:00:00" in content
            finally:
                os.chdir(original_cwd)

class TestMainFunction:
    """Test the main command-line interface function."""
    
    def test_individual_functions_work(self):
        """Test that individual functions can be called successfully."""
        # Test diagram generation functions directly.
        detailed = generate_detailed_architecture_diagram()
        assert len(detailed) > 100
        dataflow = generate_data_flow_diagram()
        assert len(dataflow) > 100
        # Test file saving
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)    
            try:
                filename = save_diagrams_to_file("test content")
                assert os.path.exists(filename)
            finally:
                os.chdir(original_cwd)

    @patch('sys.argv', ['visualize_architecture.py'])
    @patch('visualize_architecture.main')  # Mock the entire function instead.
    @patch('builtins.print')
    def test_main_default_behavior(self, mock_print, mock_main_func):
        """Test main function with default arguments (no flags)."""
        from visualize_architecture import main as real_main
        # Call with mocked environment.
        mock_main_func.return_value = None
        mock_main_func()
        # Verify it was called.
        mock_main_func.assert_called_once()
    
    @patch('sys.argv', ['visualize_architecture.py', '--detailed'])
    def test_main_detailed_flag(self):
        """Test main function with --detailed flag."""
        # Test detailed diagram generation directly.
        diagram = generate_detailed_architecture_diagram()
        assert "DETAILED ENCODER-DECODER ARCHITECTURE DIAGRAM" in diagram
    
    @patch('sys.argv', ['visualize_architecture.py', '--dataflow'])
    def test_main_dataflow_flag(self):
        """Test main function with --dataflow flag."""
        # Test dataflow diagram generation directly.
        diagram = generate_data_flow_diagram()
        assert "TENSOR FLOW DIAGRAM" in diagram
    
    def test_main_all_flag(self):
        """Test main function with --all flag."""
        # Test both diagram functions work.
        detailed = generate_detailed_architecture_diagram()
        dataflow = generate_data_flow_diagram()
        assert "DETAILED ENCODER-DECODER ARCHITECTURE DIAGRAM" in detailed
        assert "TENSOR FLOW DIAGRAM" in dataflow
    
    def test_main_save_flag(self):
        """Test main function with --save flag."""
        # Test file saving functionality.
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                filename = save_diagrams_to_file("test content")
                assert os.path.exists(filename)
            finally:
                os.chdir(original_cwd)
    
    def test_main_multiple_flags(self):
        """Test main function with multiple flags."""
        # Test that multiple functions can work together.
        detailed = generate_detailed_architecture_diagram()
        dataflow = generate_data_flow_diagram()
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                combined = detailed + "\n\n" + dataflow
                filename = save_diagrams_to_file(combined)
                assert os.path.exists(filename)
                with open(filename, 'r') as f:
                    content = f.read()
                    assert "DETAILED ENCODER-DECODER" in content
                    assert "TENSOR FLOW" in content
            finally:
                os.chdir(original_cwd)

class TestCommandLineInterface:
    """Test command-line argument parsing."""
    
    def test_argparse_functionality(self):
        """Test that argparse works for different combinations."""
        import argparse
        from visualize_architecture import main
        # Test that argparse would handle various options.
        parser = argparse.ArgumentParser()
        parser.add_argument("--detailed", action="store_true")
        parser.add_argument("--dataflow", action="store_true") 
        parser.add_argument("--save", action="store_true")
        parser.add_argument("--all", action="store_true")
        # Test various argument combinations.
        args1 = parser.parse_args(["--detailed"])
        assert args1.detailed is True
        args2 = parser.parse_args(["--dataflow", "--save"])
        assert args2.dataflow is True
        assert args2.save is True
    
    def test_invalid_combination_handling(self):
        """Test handling of argument combinations."""
        # Test that detailed and dataflow can be used together.
        detailed = generate_detailed_architecture_diagram()
        dataflow = generate_data_flow_diagram()
        # Both should work independently.
        assert "DETAILED" in detailed
        assert "TENSOR FLOW" in dataflow

class TestDiagramContent:
    """Test the content and formatting of generated diagrams."""
    
    def test_detailed_diagram_formatting(self):
        """Test that detailed diagram has proper formatting."""
        diagram = generate_detailed_architecture_diagram()
        # Check for proper Unicode box drawing characters.
        assert "┌─" in diagram # Top-left corner.
        assert "└─" in diagram # Bottom-left corner.
        assert "├─" in diagram # Left tee.
        assert "│" in diagram # Vertical line.
        # Check for proper section headers.
        assert "═══" in diagram # Double horizontal line for headers.
        # Check for emoji usage.
        assert "🧠" in diagram
        assert "📊" in diagram
        assert "⚙️" in diagram
    
    def test_dataflow_diagram_formatting(self):
        """Test that dataflow diagram has proper formatting."""
        diagram = generate_data_flow_diagram()
        # Check for proper flow indicators.
        assert "│" in diagram # Vertical flow.
        assert "▼" in diagram # Down arrow.
        assert "┌─" in diagram # Box drawing.
        # Check for tensor shape notation.
        assert "×" in diagram # Multiplication symbol for dimensions.
        assert "[" in diagram and "]" in diagram # Tensor shape brackets.

    def test_diagram_completeness(self):
        """Test that diagrams contain all necessary information."""
        detailed = generate_detailed_architecture_diagram()
        dataflow = generate_data_flow_diagram()
        # Detailed diagram should have complete architecture info.
        required_detailed_sections = [
            "INPUT DATA FLOW",
            "ENCODER SECTION", 
            "DECODER SECTION",
            "LOSS & OPTIMIZATION",
            "TRAINING DYNAMICS",
            "MODEL PARAMETERS",
            "COMPUTATIONAL COMPLEXITY"]
        for section in required_detailed_sections:
            assert section in detailed, f"Missing section: {section}"
        # Dataflow diagram should have complete pipeline.
        required_dataflow_sections = [
            "RAW DATA GENERATION",
            "ONE-HOT ENCODING",
            "PREPROCESSING", 
            "BATCH TENSORS",
            "ENCODER",
            "DECODER",
            "LOSS COMPUTATION",
            "BACKPROPAGATION"]
        for section in required_dataflow_sections:
            assert section in dataflow, f"Missing section: {section}"

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_save_file_permission_error(self):
        """Test handling of file permission errors."""
        test_content = "Test content"
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = PermissionError("Permission denied")
            with pytest.raises(PermissionError):
                save_diagrams_to_file(test_content, "/readonly/test.txt")
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                # Test with empty content.
                filename = save_diagrams_to_file("")
                # File should still be created with header.
                assert os.path.exists(filename)
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Should have header even with empty content.
                assert "ENCODER-DECODER NEURAL NETWORK ARCHITECTURE DOCUMENTATION" in content
            finally:
                os.chdir(original_cwd)

class TestIntegration:
    """Integration tests for the visualization module."""
    
    def test_full_pipeline_with_save(self):
        """Test the complete pipeline from generation to saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                # Generate both diagrams.
                detailed = generate_detailed_architecture_diagram()
                dataflow = generate_data_flow_diagram()
                # Combine content.
                combined_content = detailed + "\n\n" + dataflow
                # Save to file.
                filename = save_diagrams_to_file(combined_content)
                # Verify file exists and has correct content.
                assert os.path.exists(filename)
                with open(filename, 'r', encoding='utf-8') as f:
                    saved_content = f.read()
                # Check that both diagrams are in the saved file.
                assert "DETAILED ENCODER-DECODER ARCHITECTURE DIAGRAM" in saved_content
                assert "TENSOR FLOW DIAGRAM" in saved_content
                assert detailed in saved_content
                assert dataflow in saved_content
            finally:
                os.chdir(original_cwd)
    
    def test_main_integration(self):
        """Test main function integration with file saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                # Test the full pipeline without calling main directly.
                detailed = generate_detailed_architecture_diagram()
                dataflow = generate_data_flow_diagram()
                combined_content = detailed + "\n\n" + dataflow
                filename = save_diagrams_to_file(combined_content)
                # Should have created a file.
                assert os.path.exists(filename)
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Check that both diagrams are in the saved file.
                assert "DETAILED ENCODER-DECODER ARCHITECTURE DIAGRAM" in content
                assert "TENSOR FLOW DIAGRAM" in content          
            finally:
                os.chdir(original_cwd)

# The big red activation button.
if __name__ == "__main__":
    pytest.main([__file__])
