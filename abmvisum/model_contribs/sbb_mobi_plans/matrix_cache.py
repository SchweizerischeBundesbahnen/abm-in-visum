class MatrixCache:
    """
    To write.
    """

    def __init__(self, logging):
        self.logging = logging
        self.matrix_cache = {}
        self.prob_matrices = {}
        self.skims_cache = {}

    def add_skim_to_cache(self, key, values):
        self.skims_cache[key] = values

    def get_skim(self, key):
        return self.skims_cache[key]

    def add_prob_mat_to_cache(self, key, matrix):
        self.prob_matrices[key] = matrix

    def get_prob_mat(self, key):
        self.logging.info("get matrix for segment " + key + " from cache.")
        return self.prob_matrices[key]

    def has_prob_mat(self, key):
        if key in self.prob_matrices.keys():
            return True
        else:
            return False

    def add_matrix_to_cache(self, key, matrix):
        self.matrix_cache[key] = matrix

    def get_matrix(self, key):
        return self.matrix_cache[key]

    def has_matrix(self, key):
        if key in self.matrix_cache.keys():
            return True
        else:
            return False
