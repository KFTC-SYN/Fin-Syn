import unittest
import numpy as np
from tabpfgen import TabPFGen
from sklearn.datasets import make_classification, make_regression
import torch
import warnings


class TestTabPFGen(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device("cpu")
        # Use smaller values for testing to speed up execution
        self.generator = TabPFGen(
            n_sgld_steps=10,  # Reduced for testing
            sgld_step_size=0.01,
            sgld_noise_scale=0.01,
            device=self.device,
        )

        # Create synthetic datasets for testing
        self.X_class, self.y_class = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=3,
            random_state=42,
        )

        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=5, n_informative=3, noise=0.1, random_state=42
        )

        # Create imbalanced dataset for balance testing
        self.X_imbalanced, self.y_imbalanced = self._create_imbalanced_dataset()

    def _create_imbalanced_dataset(self):
        """Create an imbalanced dataset for testing balance_dataset method."""
        # Create base dataset
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=4,
            random_state=42,
        )

        # Create imbalanced classes: [100, 50, 20, 10]
        class_sizes = [100, 50, 20, 10]
        indices = []

        for class_idx, size in enumerate(class_sizes):
            class_indices = np.where(y == class_idx)[0][:size]
            indices.extend(class_indices)

        indices = np.array(indices)
        return X[indices], y[indices]

    def test_initialization(self):
        """Test proper initialization of TabPFGen."""
        self.assertEqual(self.generator.n_sgld_steps, 10)
        self.assertEqual(self.generator.sgld_step_size, 0.01)
        self.assertEqual(self.generator.sgld_noise_scale, 0.01)
        self.assertEqual(self.generator.device.type, "cpu")
        self.assertIsNotNone(self.generator.scaler)

    def test_classification_generation(self):
        """Test generation of synthetic classification data."""
        n_samples = 51  # Changed to be divisible by 3 classes
        X_synth, y_synth = self.generator.generate_classification(
            self.X_class, self.y_class, n_samples, balance_classes=True
        )

        # Check shapes
        expected_samples = (
            n_samples // 3
        ) * 3  # Round down to nearest multiple of num_classes
        self.assertEqual(X_synth.shape[0], expected_samples)
        self.assertEqual(X_synth.shape[1], self.X_class.shape[1])
        self.assertEqual(y_synth.shape[0], expected_samples)

        # Check data types
        self.assertTrue(isinstance(X_synth, np.ndarray))
        self.assertTrue(isinstance(y_synth, np.ndarray))

        # Check if generated classes are valid
        unique_classes = np.unique(self.y_class)
        self.assertTrue(all(cls in unique_classes for cls in np.unique(y_synth)))

    def test_regression_generation(self):
        """Test generation of synthetic regression data."""
        n_samples = 50
        X_synth, y_synth = self.generator.generate_regression(
            self.X_reg, self.y_reg, n_samples, use_quantiles=True
        )

        # Check shapes
        self.assertEqual(X_synth.shape[0], n_samples)
        self.assertEqual(X_synth.shape[1], self.X_reg.shape[1])
        self.assertEqual(y_synth.shape[0], n_samples)

        # Check data types
        self.assertTrue(isinstance(X_synth, np.ndarray))
        self.assertTrue(isinstance(y_synth, np.ndarray))

        # Check if generated values are within reasonable bounds
        y_min, y_max = self.y_reg.min(), self.y_reg.max()
        margin = (y_max - y_min) * 0.2  # Allow 20% margin
        self.assertTrue(np.all(y_synth >= y_min - margin))
        self.assertTrue(np.all(y_synth <= y_max + margin))

    def test_balance_dataset_default(self):
        """Test balance_dataset with default parameters (majority class balancing)."""
        # Suppress print statements during testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_synthetic, y_synthetic, X_combined, y_combined = (
                self.generator.balance_dataset(self.X_imbalanced, self.y_imbalanced)
            )

        # Check output types
        self.assertTrue(isinstance(X_synthetic, np.ndarray))
        self.assertTrue(isinstance(y_synthetic, np.ndarray))
        self.assertTrue(isinstance(X_combined, np.ndarray))
        self.assertTrue(isinstance(y_combined, np.ndarray))

        # Check feature dimensions
        self.assertEqual(X_synthetic.shape[1], self.X_imbalanced.shape[1])
        self.assertEqual(X_combined.shape[1], self.X_imbalanced.shape[1])

        # Check that synthetic samples were generated
        self.assertGreater(len(X_synthetic), 0)
        self.assertEqual(len(X_synthetic), len(y_synthetic))

        # Check that combined dataset is larger than original
        self.assertGreater(len(X_combined), len(self.X_imbalanced))
        self.assertEqual(len(X_combined), len(y_combined))

        # Check that balancing improved class distribution
        original_unique, original_counts = np.unique(
            self.y_imbalanced, return_counts=True
        )
        combined_unique, combined_counts = np.unique(y_combined, return_counts=True)

        # Calculate coefficient of variation (CV) - lower is more balanced
        original_cv = np.std(original_counts) / np.mean(original_counts)
        combined_cv = np.std(combined_counts) / np.mean(combined_counts)

        # Balancing should reduce the coefficient of variation
        self.assertLess(
            combined_cv,
            original_cv,
            "Dataset should be more balanced after synthetic generation",
        )

    def test_balance_dataset_custom_target(self):
        """Test balance_dataset with custom target per class."""
        target_size = 120

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_synthetic, y_synthetic, X_combined, y_combined = (
                self.generator.balance_dataset(
                    self.X_imbalanced, self.y_imbalanced, target_per_class=target_size
                )
            )

        # Check that balancing improved class distribution
        original_unique, original_counts = np.unique(
            self.y_imbalanced, return_counts=True
        )
        combined_unique, combined_counts = np.unique(y_combined, return_counts=True)

        # Check that we have more total samples
        self.assertGreater(len(y_combined), len(self.y_imbalanced))

        # Check that class distribution is more balanced
        original_cv = np.std(original_counts) / np.mean(original_counts)
        combined_cv = np.std(combined_counts) / np.mean(combined_counts)

        # Balancing should reduce coefficient of variation
        self.assertLess(
            combined_cv,
            original_cv,
            "Dataset should be more balanced after synthetic generation",
        )

        # Check that the largest class didn't shrink significantly
        original_max = np.max(original_counts)
        combined_max = np.max(combined_counts)
        self.assertGreaterEqual(
            combined_max, original_max, "Largest class should not shrink"
        )

    def test_balance_dataset_min_class_size(self):
        """Test balance_dataset with minimum class size filtering."""
        # Create dataset where some classes will be below threshold
        min_threshold = 15

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_synthetic, y_synthetic, X_combined, y_combined = (
                self.generator.balance_dataset(
                    self.X_imbalanced, self.y_imbalanced, min_class_size=min_threshold
                )
            )

        # Check that small classes were skipped from synthetic generation
        original_unique, original_counts = np.unique(
            self.y_imbalanced, return_counts=True
        )
        valid_classes = original_unique[original_counts >= min_threshold]

        # Should have generated synthetic samples (unless all classes are small)
        if len(valid_classes) > 0:
            self.assertGreater(len(X_synthetic), 0)

            # Check that we have improvement in balance for valid classes
            original_valid_counts = original_counts[original_counts >= min_threshold]
            if len(original_valid_counts) > 1:  # Only test if multiple valid classes
                original_cv = np.std(original_valid_counts) / np.mean(
                    original_valid_counts
                )

                combined_unique, combined_counts = np.unique(
                    y_combined, return_counts=True
                )
                combined_valid_counts = []
                for cls in valid_classes:
                    if cls in combined_unique:
                        idx = np.where(combined_unique == cls)[0][0]
                        combined_valid_counts.append(combined_counts[idx])

                if len(combined_valid_counts) > 1:
                    combined_cv = np.std(combined_valid_counts) / np.mean(
                        combined_valid_counts
                    )
                    self.assertLessEqual(
                        combined_cv,
                        original_cv + 0.1,
                        "Valid classes should be more balanced",
                    )

    def test_balance_dataset_no_balancing_needed(self):
        """Test balance_dataset when dataset is already balanced."""
        # Create balanced dataset with correct parameters
        X_balanced, y_balanced = make_classification(
            n_samples=120,
            n_features=4,
            n_classes=3,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            class_sep=1.0,
            random_state=42,
        )

        # Manually ensure perfect balance
        samples_per_class = 40
        balanced_indices = []
        for class_idx in range(3):
            class_indices = np.where(y_balanced == class_idx)[0][:samples_per_class]
            balanced_indices.extend(class_indices)

        X_balanced = X_balanced[balanced_indices]
        y_balanced = y_balanced[balanced_indices]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_synthetic, y_synthetic, X_combined, y_combined = (
                self.generator.balance_dataset(X_balanced, y_balanced)
            )

        # Should return empty synthetic arrays
        self.assertEqual(len(X_synthetic), 0)
        self.assertEqual(len(y_synthetic), 0)

        # Combined should be same as original
        np.testing.assert_array_equal(X_combined, X_balanced)
        np.testing.assert_array_equal(y_combined, y_balanced)

    def test_balance_dataset_input_validation(self):
        """Test input validation for balance_dataset method."""
        # Test mismatched array lengths
        with self.assertRaises(ValueError):
            self.generator.balance_dataset(self.X_imbalanced[:-1], self.y_imbalanced)

        # Test target_per_class less than maximum class size
        max_class_size = np.max(np.unique(self.y_imbalanced, return_counts=True)[1])
        with self.assertRaises(ValueError):
            self.generator.balance_dataset(
                self.X_imbalanced,
                self.y_imbalanced,
                target_per_class=max_class_size - 1,
            )

    def test_balance_dataset_all_classes_too_small(self):
        """Test balance_dataset when all classes are below minimum threshold."""
        # Create dataset with all small classes
        X_small, y_small = make_classification(
            n_samples=20,
            n_features=4,
            n_classes=2,  # Reduced to 2 classes to avoid sklearn constraint
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42,
        )

        with self.assertRaises(ValueError):
            self.generator.balance_dataset(
                X_small,
                y_small,
                min_class_size=15,  # All classes will be below this
            )

    def test_generate_samples_for_class(self):
        """Test the helper method _generate_samples_for_class."""
        # Prepare data
        X_scaled = self.generator.scaler.fit_transform(self.X_imbalanced)
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(self.y_imbalanced, device=self.device)

        # Test sample generation for a specific class
        class_label = 0
        n_samples = 10

        x_synth = self.generator._generate_samples_for_class(
            class_label, n_samples, x_train, y_train, X_scaled
        )

        # Check output
        self.assertEqual(x_synth.shape[0], n_samples)
        self.assertEqual(x_synth.shape[1], X_scaled.shape[1])
        self.assertTrue(torch.is_tensor(x_synth))

    def test_balance_dataset_reproducibility(self):
        """Test reproducibility of balance_dataset with fixed seeds."""
        # Set seeds
        np.random.seed(42)
        torch.manual_seed(42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_syn1, y_syn1, X_comb1, y_comb1 = self.generator.balance_dataset(
                self.X_imbalanced, self.y_imbalanced
            )

        # Reset seeds
        np.random.seed(42)
        torch.manual_seed(42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_syn2, y_syn2, X_comb2, y_comb2 = self.generator.balance_dataset(
                self.X_imbalanced, self.y_imbalanced
            )

        # Check reproducibility
        np.testing.assert_array_almost_equal(X_syn1, X_syn2, decimal=5)
        np.testing.assert_array_equal(y_syn1, y_syn2)
        np.testing.assert_array_almost_equal(X_comb1, X_comb2, decimal=5)
        np.testing.assert_array_equal(y_comb1, y_comb2)

    def test_energy_computation(self):
        """Test the energy computation function."""
        x_synth = torch.randn(10, 5, device=self.device)
        y_synth = torch.randint(0, 3, (10,), device=self.device)
        x_train = torch.randn(20, 5, device=self.device)
        y_train = torch.randint(0, 3, (20,), device=self.device)

        energy = self.generator._compute_energy(x_synth, y_synth, x_train, y_train)

        # Check shape and type
        self.assertEqual(energy.shape, (10,))
        self.assertTrue(torch.is_tensor(energy))

        # Check if energy is non-negative
        self.assertTrue(torch.all(energy >= 0))

    def test_sgld_step(self):
        """Test the SGLD step function."""
        x_synth = torch.randn(10, 5, device=self.device)
        y_synth = torch.randint(0, 3, (10,), device=self.device)
        x_train = torch.randn(20, 5, device=self.device)
        y_train = torch.randint(0, 3, (20,), device=self.device)

        x_new = self.generator._sgld_step(x_synth, y_synth, x_train, y_train)

        # Check shape and type
        self.assertEqual(x_new.shape, x_synth.shape)
        self.assertTrue(torch.is_tensor(x_new))

        # Check if values have changed
        self.assertFalse(torch.allclose(x_new, x_synth))

    def test_edge_cases(self):
        """Test edge cases and potential error conditions."""
        # Test with single sample
        X_single = self.X_class[:1]
        y_single = self.y_class[:1]

        with self.assertRaises(ValueError):
            # Should raise error for n_samples > number of training samples
            self.generator.generate_classification(
                X_single, y_single, 2, balance_classes=True
            )

        # Test with zero variance feature
        X_zero_var = np.copy(self.X_class)
        X_zero_var[:, 0] = 1.0

        # Should handle zero variance feature without errors
        X_synth, y_synth = self.generator.generate_classification(
            X_zero_var, self.y_class, 10, balance_classes=False
        )
        self.assertEqual(X_synth.shape[1], X_zero_var.shape[1])

    def test_reproducibility(self):
        """Test reproducibility with fixed random seed."""
        np.random.seed(42)
        torch.manual_seed(42)

        X_synth1, y_synth1 = self.generator.generate_classification(
            self.X_class, self.y_class, 20, balance_classes=True
        )

        np.random.seed(42)
        torch.manual_seed(42)

        X_synth2, y_synth2 = self.generator.generate_classification(
            self.X_class, self.y_class, 20, balance_classes=True
        )

        # Check if results are identical with same seed
        np.testing.assert_array_almost_equal(X_synth1, X_synth2)
        np.testing.assert_array_equal(y_synth1, y_synth2)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
