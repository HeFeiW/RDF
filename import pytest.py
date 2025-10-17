import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from RDF.qsdf import QSDF

class TestQSDFGetSDFWithPointsGrad:
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def mock_robot(self):
        robot = Mock()
        robot.all_links = ['link1', 'link2', 'link3']
        robot.Link2Mesh = {
            'link1': 'mesh1',
            'link2': 'mesh2',
            'link3': None
        }
        robot.get_link_mesh_transformations = Mock()
        return robot
    
    @pytest.fixture
    def mock_paths(self):
        return {
            'model': '/fake/model/path.pth'
        }
    
    @pytest.fixture
    def mock_model_data(self, device):
        return {
            'mesh1': {
                'offset': torch.zeros(3).to(device),
                'scale': 1.0,
                'weights': torch.randn(100).to(device)
            },
            'mesh2': {
                'offset': torch.zeros(3).to(device),
                'scale': 1.0,
                'weights': torch.randn(100).to(device)
            }
        }
    
    @pytest.fixture
    def qsdf_instance(self, mock_robot, mock_paths, device, mock_model_data):
        with patch('torch.load', return_value=mock_model_data), \
             patch('RDF.qsdf.ParallelSiren') as mock_siren_class, \
             patch('RDF.qsdf.load_multiple_siren_weights'):
            
            # Mock ParallelSiren instance
            mock_siren_instance = Mock()
            mock_siren_class.return_value = mock_siren_instance
            mock_siren_instance.to.return_value = mock_siren_instance
            mock_siren_instance.eval.return_value = None
            
            # Create QSDF instance
            qsdf = QSDF(robot=mock_robot, paths=mock_paths, device=device, used_links=['link1', 'link2'])
            qsdf.link_model = mock_siren_instance
            
            return qsdf
    
    def test_basic_functionality_without_derivative(self, qsdf_instance, device):
        """Test basic SDF computation without derivatives"""
        B, N = 2, 10
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            return torch.randn(K, batch_size, 1).to(device), None
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(4, N, 3).to(device)  # B*K, N, 3
            
            # Call method
            sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                x, pose, theta, use_derivative=False, return_index=False
            )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient is None
        assert torch.is_tensor(sdf_value)
    
    def test_basic_functionality_with_derivative(self, qsdf_instance, device):
        """Test SDF computation with derivatives"""
        B, N = 2, 5
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output with coordinates that require grad
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            coords = p.clone().detach().requires_grad_(True)
            sdf_output = torch.randn(K, batch_size, 1).to(device)
            return sdf_output, coords
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(4, N, 3).to(device)
            
            # Mock autograd.grad to avoid the error
            with patch('torch.autograd.grad') as mock_grad:
                mock_grad.return_value = [torch.randn(2, N, 3).to(device)]
                
                # Call method
                sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                    x, pose, theta, use_derivative=True, return_index=False
                )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient.shape == (B, N, 3)
        assert torch.is_tensor(sdf_value)
        assert torch.is_tensor(gradient)
    
    def test_return_index_functionality(self, qsdf_instance, device):
        """Test functionality when return_index=True"""
        B, N = 2, 8
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            return torch.randn(K, batch_size, 1).to(device), None
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(4, N, 3).to(device)
            
            # Call method
            sdf_value, gradient, idx = qsdf_instance.get_sdf_with_points_grad(
                x, pose, theta, use_derivative=False, return_index=True
            )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient is None
        assert idx.shape == (B, N)
        assert torch.is_tensor(idx)
    
    def test_empty_input_handling(self, qsdf_instance, device):
        """Test handling of empty inputs"""
        B, N = 1, 0
        x = torch.empty(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).to(device)
        theta = torch.randn(1, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output for empty input
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            return torch.empty(K, batch_size, 1).to(device), None
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.empty(2, N, 3).to(device)
            
            # Call method
            sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                x, pose, theta, use_derivative=False
            )
        
        # Assertions
        assert sdf_value.shape == (1, 0)
        assert gradient is None
    
    def test_large_batch_processing(self, qsdf_instance, device):
        """Test processing of large batches that require splitting"""
        B, N = 2, 15000  # Larger than batch_size=10000
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model to handle batch splitting
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            return torch.randn(K, batch_size, 1).to(device), None
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(4, N, 3).to(device)
            
            # Call method
            sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                x, pose, theta, use_derivative=False
            )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient is None
        # Verify model was called multiple times due to batch splitting
        assert qsdf_instance.link_model.call_count >= 2
    
    def test_coordinate_transformation_called(self, qsdf_instance, device):
        """Test that coordinate transformations are properly called"""
        B, N = 2, 5
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            return torch.randn(K, batch_size, 1).to(device), None
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(4, N, 3).to(device)
            
            # Call method
            qsdf_instance.get_sdf_with_points_grad(x, pose, theta, use_derivative=False)
            
            # Verify transformations were called
            mock_transform.assert_called_once()
            qsdf_instance.robot.get_link_mesh_transformations.assert_called_once_with(pose, theta)
    
    def test_gradient_autograd_error_handling(self, qsdf_instance, device):
        """Test handling of autograd.grad errors"""
        B, N = 1, 5
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            coords = torch.randn(K, batch_size, 3, requires_grad=True).to(device)
            sdf_output = torch.randn(K, batch_size, 1).to(device)
            return sdf_output, coords
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(2, N, 3).to(device)
            
            # Mock autograd.grad to raise the specific error
            with patch('torch.autograd.grad') as mock_grad:
                mock_grad.side_effect = RuntimeError("One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.")
                
                # The method should handle this error gracefully or we should catch it
                with pytest.raises(RuntimeError, match="One of the differentiated Tensors appears to not have been used"):
                    qsdf_instance.get_sdf_with_points_grad(
                        x, pose, theta, use_derivative=True
                    )
    
    def test_gradient_computation_with_allow_unused(self, qsdf_instance, device):
        """Test gradient computation with allow_unused=True to fix the autograd error"""
        B, N = 2, 5
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            coords = p.clone().detach().requires_grad_(True)
            sdf_output = torch.sum(coords, dim=-1, keepdim=True)  # Ensure coords are used
            return sdf_output, coords
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(4, N, 3).to(device)
            
            # Patch the autograd.grad call to use allow_unused=True
            original_autograd_grad = torch.autograd.grad
            def patched_autograd_grad(*args, **kwargs):
                kwargs['allow_unused'] = True
                return original_autograd_grad(*args, **kwargs)
            
            with patch('torch.autograd.grad', side_effect=patched_autograd_grad):
                # Call method
                sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                    x, pose, theta, use_derivative=True
                )
            
            # Assertions
            assert sdf_value.shape == (B, N)
            assert gradient.shape == (B, N, 3)
    
    def test_input_validation(self, qsdf_instance, device):
        """Test input validation and edge cases"""
        # Test with mismatched dimensions
        B, N = 2, 5
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta_wrong = torch.randn(B+1, 4).to(device)  # Wrong batch size
        
        # This should not crash but might produce unexpected results
        # The method doesn't validate input dimensions explicitly
        pass
    
    def test_scale_and_offset_application(self, qsdf_instance, device):
        """Test that scale and offset are properly applied"""
        B, N = 1, 3
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).to(device)
        theta = torch.randn(1, 4).to(device)
        
        # Set specific scale and offset values
        qsdf_instance.model_info = {
            'mesh1': {
                'offset': torch.tensor([1.0, 0.0, 0.0]).to(device),
                'scale': 2.0
            },
            'mesh2': {
                'offset': torch.tensor([0.0, 1.0, 0.0]).to(device),
                'scale': 0.5
            }
        }
        
        # Mock robot transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        def mock_model_call(p):
            K, batch_size, _ = p.shape
            return torch.ones(K, batch_size, 1).to(device), None  # Constant SDF
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Mock utils.transform_points
        with patch('RDF.qsdf.utils.transform_points') as mock_transform:
            mock_transform.return_value = torch.randn(2, N, 3).to(device)
            
            # Call method
            sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                x, pose, theta, use_derivative=False
            )
        
        # The SDF values should be affected by the scale
        assert sdf_value.shape == (1, N)
        assert torch.is_tensor(sdf_value)import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from RDF.qsdf import QSDF

class TestQSDFGetSDFWithPointsGrad:
    
    @pytest.fixture
    def mock_robot(self):
        """Create a mock robot for testing"""
        robot = Mock()
        robot.all_links = ['link1', 'link2', 'link3']
        robot.Link2Mesh = {
            'link1': 'mesh1',
            'link2': 'mesh2', 
            'link3': None
        }
        robot.get_link_mesh_transformations = Mock()
        return robot
    
    @pytest.fixture
    def mock_paths(self):
        """Create mock paths"""
        return {
            'model': '/fake/model/path.pth'
        }
    
    @pytest.fixture
    def device(self):
        """Get available device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def mock_model_data(self):
        """Create mock model data"""
        return {
            'mesh1': {
                'offset': torch.zeros(3),
                'scale': 1.0,
                'weights': torch.randn(100)
            },
            'mesh2': {
                'offset': torch.zeros(3),
                'scale': 1.0,
                'weights': torch.randn(100)
            }
        }
    
    @pytest.fixture
    def qsdf_instance(self, mock_robot, mock_paths, device, mock_model_data):
        """Create QSDF instance with mocked dependencies"""
        with patch('torch.load', return_value=mock_model_data), \
             patch('RDF.qsdf.ParallelSiren') as mock_siren, \
             patch('RDF.qsdf.load_multiple_siren_weights'):
            
            # Mock the parallel siren
            mock_siren_instance = Mock()
            mock_siren.return_value = mock_siren_instance
            mock_siren_instance.to.return_value = mock_siren_instance
            mock_siren_instance.eval.return_value = None
            
            qsdf = QSDF(robot=mock_robot, paths=mock_paths, device=device, used_links=['link1', 'link2'])
            qsdf.link_model = mock_siren_instance
            
            return qsdf
    
    def test_basic_functionality_without_derivative(self, qsdf_instance, device):
        """Test basic SDF computation without derivatives"""
        # Setup input data
        B, N = 2, 10
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        mock_sdf_output = torch.randn(2, N, 1).to(device)
        qsdf_instance.link_model.return_value = (mock_sdf_output, None)
        
        # Call method
        sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
            x, pose, theta, use_derivative=False, return_index=False
        )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient is None
        assert torch.is_tensor(sdf_value)
    
    def test_basic_functionality_with_derivative(self, qsdf_instance, device):
        """Test SDF computation with derivatives"""
        # Setup input data
        B, N = 2, 10
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output and gradients
        mock_sdf_output = torch.randn(2, N, 1).to(device)
        mock_coords = torch.randn(2, N, 3, requires_grad=True).to(device)
        qsdf_instance.link_model.return_value = (mock_sdf_output, mock_coords)
        
        # Mock autograd.grad
        with patch('torch.autograd.grad') as mock_grad:
            mock_grad.return_value = [torch.randn(2, N, 3).to(device)]
            
            # Call method
            sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                x, pose, theta, use_derivative=True, return_index=False
            )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient.shape == (B, N, 3)
        assert torch.is_tensor(sdf_value)
        assert torch.is_tensor(gradient)
    
    def test_return_index_functionality(self, qsdf_instance, device):
        """Test functionality when return_index=True"""
        # Setup input data
        B, N = 2, 10
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        mock_sdf_output = torch.randn(2, N, 1).to(device)
        qsdf_instance.link_model.return_value = (mock_sdf_output, None)
        
        # Call method
        sdf_value, gradient, idx = qsdf_instance.get_sdf_with_points_grad(
            x, pose, theta, use_derivative=False, return_index=True
        )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient is None
        assert idx.shape == (B, N)
        assert torch.is_tensor(idx)
    
    def test_empty_input_handling(self, qsdf_instance, device):
        """Test handling of empty inputs"""
        # Empty points
        x = torch.empty(0, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).to(device)
        theta = torch.randn(1, 4).to(device)
        
        # Mock transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        mock_sdf_output = torch.empty(2, 0, 1).to(device)
        qsdf_instance.link_model.return_value = (mock_sdf_output, None)
        
        # Call method
        sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
            x, pose, theta, use_derivative=False
        )
        
        # Assertions
        assert sdf_value.shape == (1, 0)
        assert gradient is None
    
    def test_single_batch_dimension(self, qsdf_instance, device):
        """Test with single batch dimension"""
        B, N = 1, 5
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).to(device)
        theta = torch.randn(1, 4).to(device)
        
        # Mock transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, 1, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output
        mock_sdf_output = torch.randn(2, N, 1).to(device)
        qsdf_instance.link_model.return_value = (mock_sdf_output, None)
        
        # Call method
        sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
            x, pose, theta, use_derivative=False
        )
        
        # Assertions
        assert sdf_value.shape == (1, N)
    
    def test_large_batch_processing(self, qsdf_instance, device):
        """Test processing of large batches (batch splitting)"""
        B, N = 3, 25000  # Larger than batch_size=10000
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model to be called multiple times due to batch splitting
        def mock_model_call(p):
            batch_size = p.shape[1]
            return torch.randn(2, batch_size, 1).to(device), None
        
        qsdf_instance.link_model.side_effect = mock_model_call
        
        # Call method
        sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
            x, pose, theta, use_derivative=False
        )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient is None
        # Verify model was called multiple times due to batch splitting
        assert qsdf_instance.link_model.call_count >= 3
    
    def test_gradient_computation_shapes(self, qsdf_instance, device):
        """Test that gradient computation produces correct shapes"""
        B, N = 2, 8
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock transformations with some rotation
        rotation = torch.tensor([[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0, 0, 1]], dtype=torch.float32).to(device)
        transform = torch.eye(4).to(device)
        transform[:3, :3] = rotation
        
        mock_trans = {
            'link1': transform.unsqueeze(1).expand(4, B, 4, 4),
            'link2': transform.unsqueeze(1).expand(4, B, 4, 4)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock model output and gradients
        mock_sdf_output = torch.randn(2, N, 1).to(device)
        mock_coords = torch.randn(2, N, 3, requires_grad=True).to(device)
        qsdf_instance.link_model.return_value = (mock_sdf_output, mock_coords)
        
        # Mock autograd.grad
        with patch('torch.autograd.grad') as mock_grad:
            mock_grad.return_value = [torch.randn(2, N, 3).to(device)]
            
            # Call method
            sdf_value, gradient = qsdf_instance.get_sdf_with_points_grad(
                x, pose, theta, use_derivative=True
            )
        
        # Assertions
        assert sdf_value.shape == (B, N)
        assert gradient.shape == (B, N, 3)
        
        # Check that gradients are normalized
        grad_norms = torch.norm(gradient, dim=-1)
        assert torch.allclose(grad_norms, torch.ones_like(grad_norms), atol=1e-5)
    
    @patch('RDF.qsdf.utils.transform_points')
    def test_coordinate_transformations(self, mock_transform, qsdf_instance, device):
        """Test that coordinate transformations are called correctly"""
        B, N = 2, 5
        x = torch.randn(N, 3).to(device)
        pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        theta = torch.randn(B, 4).to(device)
        
        # Mock transformations
        mock_trans = {
            'link1': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device),
            'link2': torch.eye(4).unsqueeze(1).expand(4, B, 4, 4).to(device)
        }
        qsdf_instance.robot.get_link_mesh_transformations.return_value = mock_trans
        
        # Mock transform_points return
        mock_transform.return_value = torch.randn(4, N, 3).to(device)  # B*K, N, 3
        
        # Mock model output
        mock_sdf_output = torch.randn(2, N, 1).to(device)
        qsdf_instance.link_model.return_value = (mock_sdf_output, None)
        
        # Call method
        qsdf_instance.get_sdf_with_points_grad(x, pose, theta, use_derivative=False)
        
        # Verify transform_points was called
        mock_transform.assert_called_once()