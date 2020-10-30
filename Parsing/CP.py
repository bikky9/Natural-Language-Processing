from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
)
prediction = predictor.predict(
    sentence="The agile athlete runs fast and jumps high"
)
print(prediction['trees'])
