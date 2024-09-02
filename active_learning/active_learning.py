import numpy as np

class BatchActiveLearning():
    def __init__(self, b_samples):
        self.b_samples = b_samples
        # print(f'Getting {self.b_samples} samples')

    def get_u_function(self, x_mc_norm, mean_prediction, std_prediction):
        
        u_function = (np.abs(mean_prediction))/std_prediction
        selected_indices = np.argsort(u_function)[:self.b_samples]
        # Select the corresponding samples from x_mc_norm
        selected_samples_norm = x_mc_norm[selected_indices]

        return selected_samples_norm

    # def get_correlation(self, x, x_mc, y_mc, n_points):
    #     y_mean, y_std = torch.mean(y_mc, 1), torch.std(y_mc, 1)
    #     u_function = -(y_mean.abs())/y_std
    #     sorted = torch.topk(u_function, int(n_points))
    #     idx_max_ystd = sorted[1] # Taking the indices of the max std
    #     x_new = x_mc[idx_max_ystd]
    #     x_next = torch.cat( (x, x_new))
    #     return x_next
