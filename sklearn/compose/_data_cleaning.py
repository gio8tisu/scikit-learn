import warnings

import numpy as np

from ..base import BaseEstimator, clone
from ..utils.validation import check_is_fitted
from ..utils import check_array
from ..preprocessing import FunctionTransformer

__all__ = ['CleanDataEstimator']


class CleanDataEstimator(BaseEstimator):
    """Meta-estimator to fit on clean data.

    Useful for validating data cleansing steaps.

    The computation during ``fit`` is::

        clean_indices = np.nonzero(func(X, y))
        X_clean X[clean_indices]
        y_clean y[clean_indices]
        estimator.fit(X_clean, y_clean)

    The computation during ``predict`` does not apply data cleaning, i.e.::

        estimator.predict(X)

    Parameters
    ----------
    estimator : object, default=LinearRegression()
        Estimator object (i.e., Regressor/Classifier) that will be
        trained on cleaned data.

    func : function, optional
        Function that returns zeroes on "dirty"/unwanted datapoints based
        on X and/or y.
        If ``func`` is ``None``, the function used will return all indices.

    Attributes
    ----------
    estimator_ : object
        Fitted estimator.

    cleaner_ : object
        Transformer used for cleaning in ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.compose import CleanDataEstimator
    >>> clean_clf = CleanDataEstimator(estimator=LogisticRegression(),
    ...                                func=lambda X, y: 1)
    >>> X = np.arange(4).reshape(-1, 1)
    >>> y = np.exp(2 * X).ravel()
    >>> tt.fit(X, y)
    TransformedTargetRegressor(...)
    >>> tt.score(X, y)
    1.0
    >>> tt.regressor_.coef_
    array([2.])

    """
    def __init__(self, regressor=None, transformer=None,
                 func=None, inverse_func=None, check_inverse=True):
        self.regressor = regressor
        self.func = func

    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the underlying
            regressor.


        Returns
        -------
        self : object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')

        if self.regressor is None:
            from ..linear_model import LinearRegression
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        clean_idx = np.nonzero(self.func_(X, y))
        X_clean = X[clean_idx]
        y_clean = y[clean_idx]
        self.estimator_.fit(X_clean, y_clean, **fit_params)

        return self

    def predict(self, X):
        """Predict using the base regressor, applying inverse.

        The regressor is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.

        """
        check_is_fitted(self)
        pred = self.regressor_.predict(X)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(
                pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if (self._training_dim == 1 and
                pred_trans.ndim == 2 and pred_trans.shape[1] == 1):
            pred_trans = pred_trans.squeeze(axis=1)

        return pred_trans

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}
