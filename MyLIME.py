import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from scipy.stats import t


class MyLIME:
    def __init__(self, model_predict, training_data, num_samples=5000, distance_metric='euclidean'):
        self.model_predict = model_predict
        self.training_data = training_data
        self.num_samples = num_samples
        self.distance_metric = distance_metric
    
    def kernel_fn(self, distances, bandwidth=1.0):
        return np.exp(-(distances ** 2) / (bandwidth ** 2))
    
    def perturb_instance(self, instance):
        # Perturb the instance by sampling around it
        perturbed = resample(self.training_data, n_samples=self.num_samples, replace=True)
        perturbed = np.vstack([perturbed, instance])  # Including the instance itself
        return perturbed

    def compute_distances(self, instance, perturbed_data):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(perturbed_data - instance.values.reshape(1, -1), axis=1)

    def explain(self, instance):
        # 1. Generate perturbed samples
        perturbed_data = self.perturb_instance(instance)
        
        # 2. Compute model predictions for perturbed samples
        perturbed_predictions = self.model_predict(perturbed_data)
        
        # 3. Compute distances and weights
        distances = self.compute_distances(instance, perturbed_data)
        weights = self.kernel_fn(distances)
        
        # 4. Fit a weighted linear regression model
        linear_model = LinearRegression()
        linear_model.fit(perturbed_data, perturbed_predictions, sample_weight=weights)
        
        # 5. Compute t-statistics and p-values
        predictions = linear_model.predict(perturbed_data)
        residuals = perturbed_predictions - predictions
        s_err = np.sum(np.power(residuals, 2))
        confidence_interval = t.ppf(1 - 0.025, self.num_samples - 1)
        stderr = np.sqrt((s_err/(self.num_samples - 2)) * (1.0/self.num_samples + (perturbed_data - np.mean(perturbed_data, axis=0)) ** 2 / np.sum((perturbed_data - np.mean(perturbed_data, axis=0)) ** 2, axis=0)))
        t_stats = linear_model.coef_ / stderr
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), self.num_samples - 2))
        
        return linear_model.coef_[0], t_stats[0], p_values[0]
