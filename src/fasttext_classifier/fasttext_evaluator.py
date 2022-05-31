"""
"""
import fasttext

from base.evaluator import Evaluator


class FastTextEvaluator(Evaluator):
    """ """

    def __init__(self):
        """ """
        super().__init__()

    def get_pred(self, lib: str, mod: fasttext.FastText):
        """
        Returns the prediction of model `mod` on text `lib`
        along with the output probability.

        Args:
            lib: Text description.
            mod: Model.

        Returns:
            List: List with the prediction and probability for the
                given text.
        """
        out = mod.predict(lib)
        pred = out[0][0].replace("__label__", "")
        prob = out[1][0]
        return [pred, prob]

    def evaluate(self, df_train, df_test, model):
        """ """
        # predict testing data
        df_test[["PREDICTION_NIV5", "PROBA"]] = (
            df_test["LIB_CLEAN"].apply(lambda x: self.get_pred(x, model)).to_list()
        )
        df_test["GoodPREDICTION"] = df_test["APE_NIV5"] == df_test["PREDICTION_NIV5"]
        for i in range(2, 5):
            df_test["PREDICTION_NIV" + str(i)] = df_test["PREDICTION_NIV5"].str[:i]

        # predict training data
        df_train[["PREDICTION_NIV5", "PROBA"]] = (
            df_train["LIB_CLEAN"].apply(lambda x: self.get_pred(x, model)).to_list()
        )
        df_train["GoodPREDICTION"] = df_train["APE_NIV5"] == df_train["PREDICTION_NIV5"]
        for i in range(2, 5):
            df_train["PREDICTION_NIV" + str(i)] = df_train["PREDICTION_NIV5"].str[:i]

        return df_train, df_test
