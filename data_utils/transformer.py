import scipy.sparse as sp


class Transformer(object):

    def __init__(self, flags_obj):
        self.name = flags_obj.model_type + '_transformer'


class SparseTransformer(Transformer):

    def __init__(self, flags_obj):
        super(SparseTransformer, self).__init__(flags_obj)

    def coo2lil(self, record):
        lil_record = record.tolil(copy=True)
        return lil_record

    def coo2dok(self, record):
        dok_record = record.todok(copy=True)
        return dok_record
