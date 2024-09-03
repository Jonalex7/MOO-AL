import numpy as np

class BatchActiveLearning():
    def __init__(self):
        print('engine humming...')

    def get_u_function(self, mean_prediction, std_prediction, b_samples):
        
        u_function = (np.abs(mean_prediction))/std_prediction
        selected_indices = np.argsort(u_function)[:b_samples]
        selected_indices = selected_indices.tolist()
        # Select the corresponding samples from x_mc_norm
        # selected_samples = x_mc[selected_indices]

        return selected_indices

    def get_correlation(self, x_mc, model, mean_prediction, std_prediction, b_samples):
        #firs sample evaluated with U_function
        selected_indices = self.get_u_function(mean_prediction, std_prediction, 1)

        for sample in range(b_samples-1):
            #Covariance computation
            det_cov = []
            for sample in range(len(x_mc)):
                x_assemble = x_mc[selected_indices + [sample]] 
                _, cov_assemble = model.predict(x_assemble, return_std=False, return_cov=True)
                det_ = np.linalg.det(cov_assemble)
                det_cov.append(det_)            
        
            det_cov = np.array(det_cov)

            #to avoid zeros in already selected
            det_cov[selected_indices] = 1e-9

            #evaluate U_function normalised with det_cov
            u_function = (np.abs(mean_prediction))/det_cov

            #avoid selected values
            u_function[selected_indices] = np.inf

            u_idx = np.argsort(u_function)[:1].item()
            selected_indices.append(u_idx)
            
            # print(f'{len(selected_indices)}', end=" ")

        return selected_indices
