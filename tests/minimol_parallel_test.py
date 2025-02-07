import pytest
import torch
from unittest.mock import MagicMock, patch
from minimol_parallel import MinimolParallel  # Adjust import based on actual file structure

@pytest.fixture
def mock_minimol():
    """Fixture to create a mock instance of MinimolParallel."""
    minimol = MinimolParallel(batch_size=10)
    minimol.datamodule = MagicMock()
    minimol.predictor = MagicMock()

    # Mock the featurization method
    def mock_featurize(smiles):
        if "invalid" in smiles[0]:
            raise ValueError("Featurization failed")  # Simulate failure
        return [{"mock_feature": torch.tensor([1.0, 2.0, 3.0])}], None

    minimol.datamodule._featurize_molecules.side_effect = mock_featurize
    return minimol


def test_featurize_batch_success(mock_minimol):
    """Test that featurization works for valid SMILES."""
    smiles_list = ["C", "CC", "CCC"]
    result = mock_minimol.featurize_batch(smiles_list)
    
    assert len(result) == len(smiles_list), "Not all SMILES were featurized!"
    assert isinstance(result[0], dict), "Featurization output is not a dictionary!"
    assert "mock_feature" in result[0], "Featurization result missing expected key!"


def test_featurize_batch_with_failures(mock_minimol):
    """Test that failed featurization does not stop the entire batch."""
    smiles_list = ["C", "invalid_SMILES", "CCC"]
    result = mock_minimol.featurize_batch(smiles_list)
    
    assert len(result) == 2, "Failed SMILES were not removed correctly!"
    assert all(isinstance(res, dict) for res in result), "Valid results should be dictionaries!"


def test_parallel_featurization(mock_minimol):
    """Test that featurization is parallelized properly."""
    with patch("concurrent.futures.ThreadPoolExecutor.map") as mock_map:
        smiles_list = ["C", "CC", "CCC"]
        mock_minimol.featurize_batch(smiles_list)
        
        mock_map.assert_called_once(), "ThreadPoolExecutor should be used for parallel execution!"


def test_dynamic_batch_size():
    """Test that batch size adapts based on memory."""
    minimol = MinimolParallel(batch_size=None)
    assert minimol.batch_size > 0, "Batch size should be dynamically estimated!"
    assert isinstance(minimol.batch_size, int), "Batch size should be an integer!"


def test_full_pipeline(mock_minimol):
    """Test that the full pipeline runs without errors."""
    smiles_list = ["C", "CC", "CCC"]
    result = mock_minimol(smiles_list)

    assert isinstance(result, list), "Output should be a list!"
    assert len(result) == len(smiles_list), "Output list length mismatch!"
    assert isinstance(result[0], torch.Tensor), "Fingerprints should be tensors!"


if __name__ == "__main__":
    pytest.main()
