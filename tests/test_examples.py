import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_ex1_example():
    """Test running ex1.py example."""
    # Import the main function
    from examples.ex1 import main

    # Run it
    main()
    # Check if outputs exist
    example_dir = Path("examples")
    assert (example_dir / "ex1_blade.vtp").exists()


def test_blend_naca_example():
    """Test running blend_naca.py example."""
    # Import the main function
    from examples.blend_naca import main

    # Run it
    main()
    # Check if outputs exist
    example_dir = Path("examples")
    assert (example_dir / "planform.png").exists()
    assert (example_dir / "blended_naca.png").exists()
    assert (example_dir / "blended_naca_blade.vtp").exists()
