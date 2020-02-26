import warnings

import numpy as np

from ..base import clone
from ..utils.metaestimators import _BaseComposition
from ..utils.validation import check_is_fitted
from ..utils import check_array
from ..preprocessing import FunctionTransformer

__all__ = ['CleanDataEstimator']


class CleanDataEstimator(_BaseComposition):
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
        If ``func`` is ``None``, the used function will return all ones.

    Attributes
    ----------
    estimator_ : object
        Fitted estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.compose import CleanDataEstimator
    >>> # Consider all data-points as clean.
    >>> clean_clf = CleanDataEstimator()
    >>> X = np.random.randn(100, 2)
    >>> X[50:] += 5
    >>> y = np.concatenate([np.zeros(50), np.ones(50)])
    >>> clean_clf.fit(X, y)
    CleanDataEstimator(...)
    >>> # Test on "trivial" test data.
    >>> assert (clean_clf.predict(np.zeros((10, 2))) == np.zeros(10)).all()
    >>> assert (clean_clf.predict(5 * np.ones((10, 2))) == np.ones(10)).all()

    """
    def __init__(self, estimator=None, func=None):
        self.estimator = estimator
        # If func is None, consider all data-points.
        self.func = func or (lambda X, y: np.ones_like(y))

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
            estimator.


        Returns
        -------
        self : object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')

        if self.estimator is None:
            from ..linear_model import LogisticRegression
            self.estimator_ = LogisticRegression()
        else:
            self.estimator_ = clone(self.estimator)

        clean_idx = np.nonzero(self.func(X, y))
        X_clean = X[clean_idx]
        y_clean = y[clean_idx]
        self.estimator_.fit(X_clean, y_clean, **fit_params)

        return self

    def predict(self, X):
        """Predict using the base estimator.

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
        pred = self.estimator_.predict(X)

        return pred

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}

