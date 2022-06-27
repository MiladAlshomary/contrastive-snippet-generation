class Argument:
    """
    Subject of analysis (SOFA) stores all related data.
    """

    def __init__(self, topic: str = None, query: str = None, arg_id: str = None, sentences: list = None,
                 snippet: list = None):
        # region content specific properties
        self.topic = topic
        self.query = query
        self.arg_id = arg_id
        self.sentences = sentences if sentences is not None else list()
        self.snippet = snippet if snippet is not None else list()
        # endregion
        # region embeddings
        self.sentence_embeddings = list()
        self.argument_embedding = list()
        # endregion
        # region contra-lexrank
        self.centrality_scores = list()
        self.argumentativeness_scores = list()
        self.context_similarity = list()
        self.excerpt = list()
        self.scores = list()
        self.excerpt_indices = list()
        # endregion
        # region cluster
        self.cohesion = list()
        self.separation = list()
        self.silhouette = list()
        # endregion
        # region evaluation properties
        self.soc_sn = 0.0
        self.soc_ex = 0.0
        self.ssc = 0.0
        # endregion

    @classmethod
    def from_json(cls, json, results=False):
        arg = Argument(topic=json['topic'],
                       query=json['query'],
                       arg_id=json['arg_id'],
                       sentences=json['sentences'],
                       snippet=json['snippet'])
        if results:
            arg.excerpt = json['excerpt']
            arg.excerpt_indices = json['excerpt_indices']
        return arg

    @classmethod
    def from_argsme_json(cls, json):
        arg = Argument(topic=json['context']['discussionTitle'],
                       query=json['context']['discussionTitle'],
                       arg_id=json['id'],
                       sentences=json['premises'][0]['sentences'])
        return arg

    def __str__(self):
        return self.sentences

    def to_json(self):
        returner = dict()
        returner['topic'] = self.topic
        returner['query'] = self.query
        returner['arg_id'] = self.arg_id
        returner['sentences'] = self.sentences
        returner['snippet'] = self.snippet
        returner['excerpt'] = self.excerpt
        returner['excerpt_indices'] = [str(i) for i in self.excerpt_indices]
        returner['soc_ex'] = str(self.soc_ex)
        returner['soc_sn'] = str(self.soc_sn)
        returner['ssc'] = str(self.ssc)
        return returner

    def to_argsme_json(self):
        returner = dict()
        returner['id'] = self.arg_id
        returner['context'] = dict()
        returner['context']['discussionTitle'] = self.query
        returner['premises'] = list()
        returner['premises'].append({
            'sentences': self.sentences,
            'excerpt': self.excerpt
        })

        return returner
