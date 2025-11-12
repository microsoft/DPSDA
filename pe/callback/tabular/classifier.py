import pandas as pd
from pe.callback.callback import Callback
from pe.constant.data import TABULAR_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.metric_item import FloatListMetricItem
from pe.logging import execution_logger
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class TabClassifier(Callback):
    """Evaluate tabular classification accuracy using a tabular classifier."""

    def __init__(self, test_data, model_name="xgboost", filter_criterion=None):
        """Constructor.

        :param test_data: The test data
        :type test_data: :py:class:`pe.data.Data`
        :param model_name: The classifier model to use, defaults to "xgboost"
        :type model_name: str, optional
        :param filter_criterion: Only computes the metric based on samples satisfying the criterion. None means no
            filtering. Defaults to None
        :type filter_criterion: dict, optional
        """
        self._test_data = test_data
        self._num_classes = len(self._test_data.metadata.label_info)
        self._model_name = model_name
        self._model = self._get_model()
        self._filter_criterion = filter_criterion
        self._filter_criterion_str = str(filter_criterion).replace(" ", "")
        self._metric_name = (
            f"tabular_classifier_{self._model_name}_filter_{self._filter_criterion_str}"
            if filter_criterion
            else f"tabular_classifier_{self._model_name}"
        )

    def _get_model(self):
        """Getting the classifier model."""
        if self._model_name == "xgboost":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError(
                    "XGBoost is not installed. Please install it using "
                    '`pip install "private-evolution[tabular] @ git+https://github.com/microsoft/DPSDA.git"`.'
                )
            if self._num_classes == 2:
                return xgb.XGBClassifier(objective="binary:logistic")
            else:
                return xgb.XGBClassifier(objective="multi:softmax", num_class=self._num_classes)
        elif self._model_name == "tabicl":
            try:
                from tabicl import TabICLClassifier
            except ImportError:
                raise ImportError(
                    "TabICLClassifier is not installed. Please install it using "
                    '`pip install "private-evolution[tabular] @ git+https://github.com/microsoft/DPSDA.git"`.'
                )
            return TabICLClassifier()
        else:
            raise ValueError(f"Unsupported classifier model: {self._model_name}")

    def _encoding(self, syn_data):
        """Encoding categorical and numerical columns.

        :param syn_data: The synthetic training data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The encoded synthetic training and test data
        :rtype: tuple[:py:class:`pe.data.Data`, :py:class:`pe.data.Data`]
        """
        feature_columns = self._test_data.metadata["feature_columns"]
        syn_df = pd.DataFrame(syn_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=feature_columns)
        test_df = pd.DataFrame(self._test_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=feature_columns)
        syn_df[LABEL_ID_COLUMN_NAME] = syn_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()
        test_df[LABEL_ID_COLUMN_NAME] = self._test_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()

        for column in feature_columns + [LABEL_ID_COLUMN_NAME]:
            merged_feature = pd.concat([syn_df[column], test_df[column]])
            if column in syn_data.metadata["cat_columns"] + [LABEL_ID_COLUMN_NAME]:
                encoder = LabelEncoder()
                encoder.fit(merged_feature.values)
                syn_df[column] = encoder.transform(syn_df[column].values)
                test_df[column] = encoder.transform(test_df[column].values)
            else:
                scaler = MinMaxScaler()
                scaler.fit(merged_feature.values.reshape(-1, 1))
                syn_df[column] = scaler.transform(syn_df[column].values.reshape(-1, 1))
                test_df[column] = scaler.transform(test_df[column].values.reshape(-1, 1))

        return syn_df, test_df

    def __call__(self, syn_data):
        """Evaluate the tabular classifier on the test set.

        :param syn_data: The synthetic training data
        :type syn_data: :py:class:`pe.data.Data`
        :return: Classification accuracy metrics
        :rtype: list[:py:class:`pe.metric_item.FloatListMetricItem`]
        """
        execution_logger.info("Evaluating tabular classifier")
        syn_data = syn_data.filter(self._filter_criterion)
        execution_logger.info(f"Number of samples after filtering: {len(syn_data.data_frame)}")
        # Encoding the synthetic training and test data
        syn_df, test_df = self._encoding(syn_data)

        X_train, y_train = syn_df.drop(LABEL_ID_COLUMN_NAME, axis=1).values, syn_df[LABEL_ID_COLUMN_NAME].values
        X_test, y_test = test_df.drop(LABEL_ID_COLUMN_NAME, axis=1).values, test_df[LABEL_ID_COLUMN_NAME].values
        self._model.fit(X_train, y_train)
        y_pred = self._model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred) * 100
        execution_logger.info(f"Tabular classifier test accuracy: {test_acc:.2f}%")
        if self._num_classes == 2:
            y_pred_proba = self._model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
            execution_logger.info(f"Tabular classifier test AUC: {auc:.2f}")
        else:
            auc = -1  # hard code, not available for multi-class classification
        f1 = f1_score(y_test, y_pred, average="macro") * 100
        execution_logger.info(f"Tabular classifier test (macro) F1 score: {f1:.2f}")

        metric_items = [
            FloatListMetricItem(name=f"{self._metric_name}_test_acc", value=[float(test_acc)]),
            FloatListMetricItem(name=f"{self._metric_name}_test_f1", value=[float(f1)]),
        ]

        if auc != -1:
            metric_items.append(FloatListMetricItem(name=f"{self._metric_name}_test_auc", value=[float(auc)]))
        execution_logger.info(f"Finished evaluating tabular classifier ({self._metric_name})")

        return metric_items
