class Model:
    best_params = None
    best_score = None
    cv = None
    model = None
    scores = None

    def __init__(self, best_params, best_score, scoring, cv, model, scores):
        """

        :param best_params: best_params
        :param best_score: best_score
        :param scoring: scoring
        :param cv: cv
        :param model: model
        :param scores: scores
        """
        self.best_params = best_params
        self.best_score = {scoring: best_score}
        self.cv = cv
        self.model = model
        self.scores = scores
