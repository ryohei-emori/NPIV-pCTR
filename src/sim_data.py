import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def generate_data_with_Z(Aucs_train, Ads_train, num_X, random_impressions):
    
    N_train = Aucs_train * Ads_train
    coefficients_X_treatment = np.random.normal(0.1, 1, num_X)
    coefficient_Z = np.random.normal(0.1, 1)
    coefficients_interaction_Z_X = np.random.normal(0.1, 1, num_X)
    intercept_treatment = np.random.normal()
    coefficients_X_outcome = np.random.normal(0.1, 1, num_X)
    intercept_outcome = np.random.normal()

    coeffs_dict = {"intercept_outcome":intercept_outcome,
                   "coefficients_X_outcome":coefficients_X_outcome,
                   "intercept_treatment":intercept_treatment,
                   "coefficients_X_treatment":coefficients_X_treatment,
                   "coefficient_Z":coefficient_Z,
                   "coefficients_interaction_Z_X":coefficients_interaction_Z_X}

    def create_dataset(random_impressions=None):
        X = []
        Z = []
        eta_values = []
        auction_ids = []

        if random_impressions is not None:
            total_rows = random_impressions
        else:
            total_rows = N_train

        Aucs = total_rows // 50
        Ads = 50

        for auc_id in range(Aucs):
            A = np.random.randn(num_X+2, num_X+2)
            cov = np.dot(A, A.T)
            cov[-2, -1] = 1e-5
            cov[-1, -2] = 1e-5
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals[eigvals < 0] = 0
            cov = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)

            features_all = np.random.multivariate_normal(np.zeros(num_X+2), cov, Ads)
            Z_auc = features_all[:, -1]
            eta_auc = features_all[:, -2]
            X_auc = features_all[:, :-2]

            Z.append(Z_auc)
            X.append(X_auc)
            eta_values.extend(eta_auc)
            auction_ids.extend([auc_id]*Ads)

        X = np.vstack(X)
        Z = np.concatenate(Z)
        eta = np.array(eta_values)

        if random_impressions is not None:
            intervention = np.ones(total_rows)
        else:
            interaction_Z_X = Z.reshape(-1, 1) * X
            eta_sqrt = np.sqrt(np.abs(eta))
            linear_prob_for_intervention = intercept_treatment + np.sum(X * coefficients_X_treatment, axis=1) + coefficient_Z * Z + np.sum(interaction_Z_X * coefficients_interaction_Z_X, axis=1) + eta
            sigmoid_prob_for_intervention = 1 / (1 + np.exp(-linear_prob_for_intervention))
            true_propensity = np.random.binomial(1, sigmoid_prob_for_intervention)
            intervention = np.zeros(N_train)
            for auc_id in range(Aucs):
                start_idx = auc_id * Ads
                end_idx = start_idx + Ads
                chosen_ad = np.argmax(true_propensity[start_idx:end_idx])
                intervention[start_idx + chosen_ad] = 1

        # Generate outcome
        linear_prob = intercept_outcome + np.sum(X * coefficients_X_outcome, axis=1)[:total_rows] + eta[:total_rows] + np.random.normal(0, 0.1, total_rows)
        sigmoid_prob = 1 / (1 + np.exp(-linear_prob))
        outcome = np.random.binomial(1, sigmoid_prob)

        if random_impressions is not None:
            df = pd.DataFrame(data={'z': Z[:total_rows], 'eta': eta[:total_rows], 'intervention': intervention, 'outcome': outcome, **{f"x_{i}": X[:, i][:total_rows] for i in range(num_X)}})
        else:
            df = pd.DataFrame(data={'auction_id': auction_ids[:total_rows], 'z': Z[:total_rows], 'eta': eta[:total_rows], 'eta_sqrt': eta_sqrt[:total_rows],'intervention': intervention, 'outcome': outcome, **{f"x_{i}": X[:, i][:total_rows] for i in range(num_X)}})
        return df

    df_train = create_dataset()
    df_val = create_dataset(random_impressions=random_impressions)
                   
    return df_train, df_val, coeffs_dict
