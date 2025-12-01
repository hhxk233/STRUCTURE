"""Canonical Correlation Analysis (CCA) related functions."""

import pickle
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import torch
from cca_zoo.linear import CCA
from scipy.linalg import sqrtm
from tqdm import trange

from src.loss.clip_loss import structure_reg
from src.utils.utils import safe_normalize


class RSMixin:
    r"""
    Adds the structure regulariser R_S to any
    *linear* cca_zoo estimator.

    Order of multiple inheritance **matters**:
        >>> class CCA_RS(RSMixin, CCA): ...
    """

    def __init__(
        self,
        *args,
        lambda_rs: float = 0.1,
        lambda_cca_coeff: float = 1e-2,
        L: int = 3,
        tau: float = 0.05,
        refine_epochs: int = 2,
        lr: float = 1e-3,
        batch_size: Optional[int] = None,
        device: str = "cpu",
        **kwargs,
    ):
        self.lambda_rs = lambda_rs
        self.lambda_cca_coeff = lambda_cca_coeff
        self.L = L
        self.tau = tau
        self.refine_epochs = refine_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        super().__init__(*args, **kwargs)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # run the parent analytic fit
        super().fit(views, y=y, **kwargs)
        # optional post-hoc refinement
        if self.lambda_rs > 0 and self.refine_epochs > 0:
            self._rs_refine(views)
        return self

    def _rs_refine(self, views):
        Xs = [torch.tensor(v, dtype=torch.float32, device=self.device) for v in views]
        Ws = [
            torch.nn.Parameter(torch.tensor(w, dtype=torch.float32, device=self.device))
            for w in self.weights_
        ]

        opt = torch.optim.Adam(Ws, lr=self.lr)
        n = Xs[0].shape[0]
        for epoch in (pbar := trange(self.refine_epochs)):
            perm = torch.randperm(n, device=self.device)
            ptr = 0
            while ptr < n:
                idx = perm[ptr : ptr + (self.batch_size or n)]
                ptr += len(idx)

                Xx, Xy = Xs
                Wx, Wy = Ws
                batch_Xx, batch_Xy = Xx[idx], Xy[idx]
                batch_Zx, batch_Zy = batch_Xx @ Wx, batch_Xy @ Wy

                batch_Zx = safe_normalize(batch_Zx, p=2, dim=1)
                batch_Zy = safe_normalize(batch_Zy, p=2, dim=1)

                r_s_x = structure_reg(
                    batch_Xx, batch_Zx, levels=self.L, temperature=self.tau
                )
                r_s_y = structure_reg(
                    batch_Xy, batch_Zy, levels=self.L, temperature=self.tau
                )
                loss_s = (r_s_x + r_s_y) / 2
                # negative trace to avoid degenerate solutions
                loss_cca = -(batch_Zx * batch_Zy).sum() / batch_Zx.shape[1]
                loss = self.lambda_rs * loss_s + self.lambda_cca_coeff * loss_cca
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_description(
                    f"Epoch: {epoch+1}, "
                    f"Loss: {loss.item():.4f}, "
                    f"R_S: {(loss_s * self.lambda_rs).item():.4f}, "
                    f"C_CCA: {(loss_cca * self.lambda_cca_coeff).item():.4f}"
                )

        # whiten of the weights
        with torch.no_grad():
            for X, W in zip(Xs, Ws):
                # covariance of the projected view
                C = (W.T @ X.T @ X @ W) / X.shape[0]  # (k,k)
                # whiten:  W <- W @ C^{−1/2}
                # eigvalsh is safe and symmetric
                d, U = torch.linalg.eigh(C)
                W @= U @ torch.diag(d.rsqrt()) @ U.T
        self.weights_ = [w.detach().cpu().numpy() for w in Ws]


class CCA_RS(RSMixin, CCA):
    # mix-in *first*
    pass


def origin_centered(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function returns the origin centered data matrix and the mean of each feature.

    Args:
        x: data matrix (n_samples, n_features)

    Returns:
        origin centered data matrix, mean of each feature
    """
    return x - np.mean(x, axis=0), np.mean(x, axis=0)


class NormalizedCCA:
    """Canonical Correlation Analysis (CCA) class which automatically zero-mean data."""

    def __init__(
        self,
        sim_dim: int,
        equal_weights: bool = False,
        use_reg: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the CCA model."""
        self.traindata1_mean = None
        self.traindata2_mean = None
        self.sim_dim = sim_dim
        self.equal_weights = equal_weights
        self.use_reg = use_reg
        self.kwargs = kwargs

    def fit_transform_train_data(
        self,
        traindata1: np.ndarray,
        traindata2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the CCA model to the training data.

        Args:
            traindata1: the first training data. shape: (num_samples, dim)
            traindata2: the second training data. shape: (num_samples, dim)

        Returns:
            traindata1: the first training data after CCA. shape: (num_samples, dim)
            traindata2: the second training data after CCA. shape: (num_samples, dim)
            corr_coeff: the correlation coefficient. shape: (dim,)
        """
        # Check the shape of the training data
        # zero mean data
        traindata1, traindata1_mean = origin_centered(traindata1)
        traindata2, traindata2_mean = origin_centered(traindata2)
        self.traindata1_mean, self.traindata2_mean = traindata1_mean, traindata2_mean
        self.traindata1, self.traindata2 = traindata1, traindata2

        # check if training data is zero-mean
        assert np.allclose(
            traindata1.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata1align not zero mean: {max(abs(traindata1.mean(axis=0)))}"
        assert np.allclose(
            traindata2.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata2align not zero mean: {max(abs(traindata2.mean(axis=0)))}"

        # CCA dimensionality reduction
        if self.use_reg:
            self.cca = CCA_RS(
                latent_dimensions=self.sim_dim,
                **self.kwargs,
            )
        else:
            self.cca = CCA(latent_dimensions=self.sim_dim)
        traindata1, traindata2 = self.cca.fit_transform((traindata1, traindata2))

        # after traindata1, traindata2 = self.cca.fit_transform(...)
        traindata1 = traindata1 / np.linalg.norm(traindata1, axis=1, keepdims=True)
        traindata2 = traindata2 / np.linalg.norm(traindata2, axis=1, keepdims=True)

        if self.equal_weights:
            corr_coeff = np.ones((traindata2.shape[1],))  # dim,
        else:
            corr_coeff = np.abs(
                np.diag(traindata1.T @ traindata2) / traindata1.shape[0]
            )  # dim,
        assert (corr_coeff <= 1 + 1e-6).all(), "Correlation should be <= 1."
        self.corr_coeff = corr_coeff
        self.traindata1, self.traindata2 = traindata1, traindata2
        return traindata1, traindata2, corr_coeff

    def transform_data(
        self,
        data1: tuple[np.ndarray, np.ndarray],
        data2: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform the data using the fitted CCA model.

        Args:
            data1: the first data. shape: (num_samples, dim)
            data2: the second data. shape: (num_samples, dim)

        Returns:
            data1: the first transformed data. shape: (num_samples, dim)
            data2: the second transformed data. shape: (num_samples, dim)
        """
        assert self.traindata1_mean is not None, "Please fit the cca model first."
        assert self.traindata2_mean is not None, "Please fit the cca model first."
        # zero mean data and transform
        data1 = data1 - self.traindata1_mean
        data2 = data2 - self.traindata2_mean
        data1, data2 = self.cca.transform((data1, data2))
        return data1, data2

    def save_model(self, path: str | Path) -> None:
        """Save the CCA class.

        Args:
            path: the path to save the class
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def load_model(self, path: str | Path) -> None:
        """Load the CCA class.

        Args:
            path: the path to load the class
        """
        if isinstance(path, str):
            path = Path(path)
        self.__dict__ = joblib.load(path.open("rb")).__dict__


class ReNormalizedCCA:
    """Canonical Correlation Analysis (CCA) class which automatically zero-mean data."""

    def __init__(self, sim_dim: int, equal_weights: bool = False) -> None:
        """Initialize the CCA model."""
        self.traindata1_mean = None
        self.traindata2_mean = None
        self.sim_dim = sim_dim
        self.equal_weights = equal_weights

    def fit_transform_train_data(
        self,
        traindata1: np.ndarray,
        traindata2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the CCA model to the training data.

        Args:
            traindata1: the first training data. shape: (num_samples, dim)
            traindata2: the second training data. shape: (num_samples, dim)

        Returns:
            traindata1: the first training data after CCA. shape: (num_samples, dim)
            traindata2: the second training data after CCA. shape: (num_samples, dim)
            corr_coeff: the correlation coefficient. shape: (dim,)
        """
        # Check the shape of the training data
        traindata1 = traindata1.astype(np.float32)
        traindata2 = traindata2.astype(np.float32)
        # zero mean data
        traindata1, traindata1_mean = origin_centered(traindata1)
        traindata2, traindata2_mean = origin_centered(traindata2)
        self.traindata1_mean, self.traindata2_mean = traindata1_mean, traindata2_mean

        # check if training data is zero-mean
        assert np.allclose(
            traindata1.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata1align not zero mean: {max(abs(traindata1.mean(axis=0)))}"
        assert np.allclose(
            traindata2.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata2align not zero mean: {max(abs(traindata2.mean(axis=0)))}"

        # CCA dimensionality reduction
        sigma_z1_inv = np.linalg.inv(
            traindata1.T @ traindata1 + np.eye(traindata1.shape[1]) * 1e-5
        )
        sigma_z1_inv_sqrt = sqrtm(sigma_z1_inv)
        sigma_z2_inv = np.linalg.inv(traindata2.T @ traindata2)
        sigma_z2_inv_sqrt = sqrtm(sigma_z2_inv)

        svd_mat = sigma_z1_inv_sqrt @ traindata1.T @ traindata2 @ sigma_z2_inv_sqrt
        u, s, vh = np.linalg.svd(svd_mat)

        self.A = u @ sigma_z1_inv_sqrt
        self.B = vh @ sigma_z2_inv_sqrt

        corr_coeff = np.ones((traindata2.shape[1],)) if self.equal_weights else s
        assert (
            corr_coeff >= 0
        ).all(), f"Correlation should be non-negative. {corr_coeff}"
        self.corr_coeff = corr_coeff
        self.traindata1, self.traindata2 = (
            (self.A @ traindata1.T).T[:, : self.sim_dim],
            (self.B @ traindata2.T).T[:, : self.sim_dim],
        )
        return self.traindata1, self.traindata2, corr_coeff

    def transform_data(
        self,
        data1: tuple[np.ndarray, np.ndarray],
        data2: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform the data using the fitted CCA model.

        Args:
            data1: the first data. shape: (num_samples, dim)
            data2: the second data. shape: (num_samples, dim)

        Returns:
            data1: the first transformed data. shape: (num_samples, dim)
            data2: the second transformed data. shape: (num_samples, dim)
        """
        data1 = data1.astype(np.float32)
        data2 = data2.astype(np.float32)
        assert self.traindata1_mean is not None, "Please fit the cca model first."
        assert self.traindata2_mean is not None, "Please fit the cca model first."
        # zero mean data and transform
        data1 = data1 - self.traindata1_mean
        data2 = data2 - self.traindata2_mean
        data1 = (self.A @ data1.T).T[:, : self.sim_dim]
        data2 = (self.B @ data2.T).T[:, : self.sim_dim]
        return data1, data2

    def save_model(self, path: str | Path) -> None:
        """Save the CCA class.

        Args:
            path: the path to save the class
        """
        if isinstance(path, str):
            path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def load_model(self, path: str | Path) -> None:
        """Load the CCA class.

        Args:
            path: the path to load the class
        """
        if isinstance(path, str):
            path = Path(path)
        self.__dict__ = joblib.load(path.open("rb")).__dict__
