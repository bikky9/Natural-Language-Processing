from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
)
prediction = predictor.predict(
    sentence="The agile athlete runs fast and jumps high "
)
print(prediction)
print(prediction['predicted_dependencies'])
