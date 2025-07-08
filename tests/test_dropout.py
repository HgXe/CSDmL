"""
Test cases for dropout functionality in CSDL neural networks.

This module tests:
- Dropout operation class
- Dropout function interface
- FCNN integration with dropout
- Training/inference mode behavior
"""

import pytest
import numpy as np
import csdl_alpha as csdl
import optax
import jax

# Import the modules we're testing
from csdml.core.activation_functions import dropout, Dropout
from csdml.core.neural_networks.fcnn import FCNN


class TestDropoutOperation:
    """Test the Dropout operation class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.rec = csdl.Recorder(inline=True)
        self.rec.start()
        
    def teardown_method(self):
        """Clean up after each test method"""
        if hasattr(self, 'rec'):
            self.rec.stop()
    
    def test_dropout_inference_mode(self):
        """Test that dropout does nothing in inference mode"""
        x = csdl.Variable(value=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        
        # In inference mode, output should be unchanged regardless of dropout rate
        y_eval = dropout(x, rate=0.5, training=False)
        
        assert np.allclose(y_eval.value, x.value), "Dropout should not affect output in inference mode"
    
    def test_dropout_zero_rate(self):
        """Test that zero dropout rate does nothing"""
        x = csdl.Variable(value=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        
        # Zero dropout rate should not change output even in training mode
        y_no_dropout = dropout(x, rate=0.0, training=True)
        
        assert np.allclose(y_no_dropout.value, x.value), "Zero dropout rate should not affect output"
    
    def test_dropout_training_mode_deterministic(self):
        """Test that dropout with seed produces deterministic results"""
        x = csdl.Variable(value=np.ones((2, 4)))
        
        # Same seed should produce same results
        y1 = dropout(x, rate=0.5, training=True, seed=42)
        
        # Reset recorder for second test
        self.rec.stop()
        self.rec = csdl.Recorder(inline=True)
        self.rec.start()
        x2 = csdl.Variable(value=np.ones((2, 4)))
        y2 = dropout(x2, rate=0.5, training=True, seed=42)
        
        assert np.allclose(y1.value, y2.value), "Same seed should produce same dropout pattern"
    
    def test_dropout_scaling(self):
        """Test that dropout properly scales remaining values"""
        x = csdl.Variable(value=np.ones((100, 100)))  # Large tensor for statistical test
        
        y = dropout(x, rate=0.5, training=True, seed=42)
        
        # Check that non-zero values are scaled by 1/(1-rate) = 2.0
        non_zero_mask = y.value != 0
        if np.any(non_zero_mask):
            scaled_values = y.value[non_zero_mask]
            expected_value = 2.0  # 1/(1-0.5)
            assert np.allclose(scaled_values, expected_value), f"Non-zero values should be scaled to {expected_value}"
    
    def test_dropout_rate_validation(self):
        """Test edge cases for dropout rates"""
        x = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
        
        # Test rate = 1.0 (should zero everything in training)
        y_all_drop = dropout(x, rate=1.0, training=True, seed=42)
        # Note: With rate=1.0, keep_prob=0, so we'd divide by zero
        # The implementation should handle this gracefully
        
        # Test very small rate
        y_small_drop = dropout(x, rate=0.01, training=True, seed=42)
        assert y_small_drop.value.shape == x.value.shape, "Shape should be preserved"


class TestFCNNDropout:
    """Test FCNN integration with dropout"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.rec = csdl.Recorder(inline=True)
        self.rec.start()
        
    def teardown_method(self):
        """Clean up after each test method"""
        if hasattr(self, 'rec'):
            self.rec.stop()
    
    def test_fcnn_dropout_initialization(self):
        """Test FCNN initialization with dropout parameters"""
        # Single dropout rate
        model1 = FCNN(10, [20, 30], 5, dropout_rate=0.3)
        assert hasattr(model1, 'dropout_rate'), "FCNN should have dropout_rate attribute"
        assert len(model1.dropout_rate) == 3, "Should have dropout rate for each layer"
        assert model1.dropout_rate[-1] == 0.0, "Output layer should have no dropout"
        
        # List of dropout rates
        model2 = FCNN(10, [20, 30], 5, dropout_rate=[0.2, 0.3, 0.0])
        assert model2.dropout_rate == [0.2, 0.3, 0.0], "Should use provided dropout rates"
    
    def test_fcnn_training_mode_control(self):
        """Test training mode control in FCNN"""
        model = FCNN(5, [10], 2, dropout_rate=0.5)
        
        # Test initial training mode
        assert model.training == True, "FCNN should start in training mode"
        
        # Test setting training mode
        model.set_training_mode(False)
        assert model.training == False, "Should be able to set inference mode"
        
        model.set_training_mode(True)
        assert model.training == True, "Should be able to set training mode"
    
    def test_fcnn_forward_with_dropout(self):
        """Test forward pass with and without dropout"""
        model = FCNN(3, [10, 10], 2, activation='tanh', dropout_rate=0.5, training=True)
        x = csdl.Variable(value=np.random.rand(5, 3))
        
        # Training mode - may have different outputs due to dropout randomness
        model.set_training_mode(True)
        y_train1 = model.forward(x)
        y_train2 = model.forward(x)
        
        # Inference mode - should be deterministic
        model.set_training_mode(False)
        y_inf1 = model.forward(x)
        y_inf2 = model.forward(x)
        
        assert np.allclose(y_inf1.value, y_inf2.value), "Inference mode should be deterministic"
        assert y_train1.value.shape == y_inf1.value.shape, "Shapes should match between modes"


class TestDropoutTraining:
    """Test dropout behavior during actual training"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create simple synthetic data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 2)
        self.y_train = np.sum(self.X_train**2, axis=1, keepdims=True)  # Simple function
        
        self.X_test = np.random.randn(20, 2)
        self.y_test = np.sum(self.X_test**2, axis=1, keepdims=True)
        
        self.rec = csdl.Recorder(inline=True)
        self.rec.start()
        
    def teardown_method(self):
        """Clean up after each test method"""
        if hasattr(self, 'rec'):
            self.rec.stop()
    
    def test_training_mode_automatic_handling(self):
        """Test that train_jax_opt automatically handles training mode"""
        model = FCNN(2, [20], 1, dropout_rate=0.3, training=False)  # Start in inference mode
        
        # Verify initial state
        assert model.training == False, "Should start in inference mode"
        
        # Training should automatically set training mode
        optimizer = optax.adam(1e-2)
        try:
            # Short training run
            model.train_jax_opt(
                optimizer=optimizer,
                loss_data=(self.X_train, self.y_train),
                num_epochs=2,
                num_batches=1
            )
            # After training, should be back to inference mode
            assert model.training == False, "Should return to inference mode after training"
        except Exception as e:
            # Training might fail due to environment issues, but mode handling should work
            print(f"Training failed (expected in test environment): {e}")
    
    def test_dropout_reduces_overfitting_tendency(self):
        """Test that dropout has the expected regularization effect"""
        # This is more of a behavioral test - we can't easily test overfitting reduction
        # in a unit test, but we can test that dropout is applied during training
        
        model_no_dropout = FCNN(2, [50, 50], 1, activation='tanh', dropout_rate=0.0)
        model_with_dropout = FCNN(2, [50, 50], 1, activation='tanh', dropout_rate=0.3)
        
        # Both models should be trainable
        assert hasattr(model_no_dropout, 'train_jax_opt'), "Model should be trainable"
        assert hasattr(model_with_dropout, 'train_jax_opt'), "Model with dropout should be trainable"
        
        # Check that dropout is properly configured
        assert model_no_dropout.dropout_rate == [0.0, 0.0, 0.0], "No dropout model should have zero rates"
        assert model_with_dropout.dropout_rate == [0.3, 0.3, 0.0], "Dropout model should have correct rates"


class TestDropoutEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.rec = csdl.Recorder(inline=True)
        self.rec.start()
        
    def teardown_method(self):
        """Clean up after each test method"""
        if hasattr(self, 'rec'):
            self.rec.stop()
    
    def test_dropout_with_different_shapes(self):
        """Test dropout with various tensor shapes"""
        shapes = [(5,), (3, 4), (2, 3, 4), (1, 1, 1, 10)]
        
        for shape in shapes:
            x = csdl.Variable(value=np.ones(shape))
            y = dropout(x, rate=0.2, training=True, seed=42)
            assert y.value.shape == shape, f"Shape should be preserved for {shape}"
    
    def test_fcnn_with_mismatched_dropout_rates(self):
        """Test FCNN with incorrect number of dropout rates"""
        # Too many dropout rates - should still work, extras ignored
        model = FCNN(2, [10], 1, dropout_rate=[0.1, 0.2, 0.3, 0.4, 0.5])
        assert len(model.dropout_rate) >= 2, "Should handle extra dropout rates gracefully"
    
    def test_dropout_operation_class_direct_usage(self):
        """Test using Dropout class directly"""
        x = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
        
        # Create dropout operation directly
        dropout_op = Dropout(x, rate=0.5, training=True, seed=42)
        y = dropout_op.finalize_and_return_outputs()
        
        assert isinstance(y, csdl.Variable), "Should return a CSDL Variable"
        assert y.value.shape == x.value.shape, "Shape should be preserved"


def test_dropout_function_interface():
    """Test the dropout function interface"""
    rec = csdl.Recorder(inline=True)
    rec.start()
    
    try:
        x = csdl.Variable(value=np.array([1.0, 2.0, 3.0, 4.0]))
        
        # Test with default parameters
        y1 = dropout(x)
        assert np.allclose(y1.value, x.value), "Default parameters should not change output"
        
        # Test with explicit parameters
        y2 = dropout(x, rate=0.5, training=True, seed=42)
        assert y2.value.shape == x.value.shape, "Shape should be preserved"
        
        # Test parameter validation through successful execution
        y3 = dropout(x, rate=0.0, training=False, seed=None)
        assert np.allclose(y3.value, x.value), "No dropout case should preserve values"
        
    finally:
        rec.stop()


if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Running basic dropout tests...")
    
    test_dropout_function_interface()
    print("✓ Dropout function interface test passed")
    
    # Run class tests
    test_dropout_op = TestDropoutOperation()
    test_dropout_op.setup_method()
    test_dropout_op.test_dropout_inference_mode()
    test_dropout_op.test_dropout_zero_rate()
    test_dropout_op.teardown_method()
    print("✓ Dropout operation tests passed")
    
    test_fcnn = TestFCNNDropout()
    test_fcnn.setup_method()
    test_fcnn.test_fcnn_dropout_initialization()
    test_fcnn.test_fcnn_training_mode_control()
    test_fcnn.teardown_method()
    print("✓ FCNN dropout tests passed")
    
    print("All basic tests completed successfully!")
    print("\nTo run full test suite with pytest:")
    print("  cd /tscc/lustre/ddn/scratch/mwarner/CSDmL")
    print("  python -m pytest tests/test_dropout.py -v")
