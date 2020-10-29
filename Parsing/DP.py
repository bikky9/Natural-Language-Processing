from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
prediction = predictor.predict(
  sentence="If I bring 10 dollars tomorrow, can you buy me lunch?"
)
print(prediction['predicted_dependencies'])