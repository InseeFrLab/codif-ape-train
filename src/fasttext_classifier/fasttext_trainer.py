"""
"""
import fasttext

from base.trainer import Trainer


class FastTextTrainer(Trainer):
    """ """

    def __init__(self):
        """ """
        super().__init__()

    def train(self, data, y_name, dim=4, epoch=20, wordNgrams=3):
        """ """
        with open("data/train_text.txt", "w") as f:
            for item in data.iterrows():
                formatted_item = "__label__{} {}".format(
                    item[1][y_name[0]], item[1]["LIB_CLEAN"]
                )
                f.write("%s\n" % formatted_item)

        model = fasttext.train_supervised(
            "data/train_text.txt", dim=dim, epoch=epoch, wordNgrams=wordNgrams
        )
        return model
