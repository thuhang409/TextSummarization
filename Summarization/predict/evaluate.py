from rouge import Rouge

class rouge_metric:
  DEFAULT_METRICS = ["rouge-1", "rouge-2", "rouge-l"]
  def __init__(self, metrics = None):
    if metrics is None:
      self.metrics = rouge_metric.DEFAULT_METRICS
    else:
      self.metrics = metrics

  def scores(self, original, predict, avg = True):
    rouge = Rouge(metrics = self.metrics)
    result = rouge.get_scores(original, predict, avg = True)
    return result
class bleu_metric:
  pass
