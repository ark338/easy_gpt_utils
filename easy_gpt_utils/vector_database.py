import os
import pinecone
import uuid
from enum import Enum
from typing import Union, List, Tuple, Optional, Dict, Any

# namespaces
class NamesSpaces(Enum):
    Glossary = 'Glossary'
    App = 'App'
    eMoped = 'eMoped'
    eScooter = 'eScooter'
    CustomService = 'CustomService'
    # etc...

def create_item(vector: list, metadata: Optional[dict] = None, id: Optional[str] = None):
    if id is None:
        id = str(uuid.uuid4())
    return (id, vector, metadata)

def create_meta(category: str, content: str, title: Optional[str] = None, url: Optional[str] = None, label: Optional[list] = None):
    kwargs = {
        'category': category,
        'content': content,
    }
    
    if title is not None:
        kwargs['title'] = title
    
    if url is not None:
        kwargs['url'] = url
    
    if label is not None:
        kwargs['label'] = label
 
    return kwargs  

# use pinecone as the first supported vector database
class Pinecone():
    def __init__(self, index: str, api_key: Optional[str]=None, environment: Optional[str]=None):
        pinecone.init(
            api_key = api_key,
            environment = environment)
        self.index = pinecone.Index(index)

    def extract_tuples(self, data_dict):
        return [vector_data for vector_data in data_dict['vectors'].values()]


    def upsert(self, namespace:str, items: Union[List[Tuple[str, List[float], Dict[str, Any]]], Tuple[str, List[float], Dict[str, Any]]]):
        self.index.upsert(items, namespace = namespace)

    # input: ids: list of id in str; namespace: str
    # output: list of dict with keys: id(str), vector(list of float), metadata(dict)
    def fetch(self, namespace: str, ids: List[str]):
        return self.extract_tuples(self.index.fetch(ids, namespace = namespace))

    # input: id: str; namespace: str
    # can update value and metadata(partially)
    def update(self, namespace: str,
               id:str,
               value: Optional[List[float]] = None, 
               metadata: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None):
        self.index.update(id, value = value, set_metadata = metadata, namespace = namespace)
    
    # input: namespace: str; ids: list of id in str; deleteAll: 'true' if delete all items in namespace
    def delete(self, namespace: str, ids: Optional[List[str]] = None, deleteAll='false'):
        self.index.delete(ids, namespace = namespace, delete_all = deleteAll)

    def query(self, 
              namespace: str, top_k: int = 5, 
              vector: Optional[List[float]] = None, 
              id: Optional[str] = None,
              filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
              include_values: Optional[bool] = None,
              include_metadata: Optional[bool] = None):
        return self.index.query(namespace = namespace, top_k = top_k, vector=vector, id=id, filter=filter, include_values=include_values, include_metadata=include_metadata)
    
    def query_meta(self, 
              namespace: str, top_k: int = 5, 
              threshold: Optional[float] = 0.0,
              vector: Optional[List[float]] = None, 
              id: Optional[str] = None,
              filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None):
        return [{'score':result['score'], 'metadata': result['metadata']} for result in 
            self.index.query(namespace = namespace, top_k = top_k, vector=vector, id=id, filter=filter, include_metadata=True)['matches'] if result['score'] > threshold]


# TODO: test code should move to unit test
# unit tests
if __name__ == "__main__":
    try:
        from .embedding import Embedding  # for when the module is imported
    except ImportError:
        from embedding import Embedding  # for when the module is run directly

    testcase = [
    'withstand voltage	耐压',
    'fingerprint U-lock	指纹U型锁',
    'No rider detected	未触发站人模式'
    ]

    use_openai = False
    if use_openai:
        # test for openai
        embedding_instance = Embedding()
    else :
        # test for azure
        embedding_instance = Embedding(
            model="text-embedding-ada-002", 
            api_type="azure", 
            api_key = os.getenv("AZURE_API_KEY"),
            api_base = "https://ninebot-rd-openai-1.openai.azure.com/",
            api_version = "2022-12-01")

    my_pinecone = Pinecone(index = 'segway-knowledge-base', environment='asia-southeast1-gcp')
    #print ("delete", my_pinecone.delete(namespace = NamesSpaces.Glossary.value, ids = ['id0']))
    #print ("my_pinecone: ", my_pinecone.fetch(namespace = NamesSpaces.Glossary.value, ids = ['id0', 'id1']))
    #print ("update test", my_pinecone.update(namespace = NamesSpaces.Glossary.value, id = 'id0', metadata = {'category': 'nihkao'}))
    #print ("delete all", my_pinecone.delete(namespace = NamesSpaces.Glossary.value, deleteAll = 'true'))

    if False:
        vectors = [vector for vector in [embedding_instance.get_raw_embedding(item) for item in testcase]]
        metas = [create_meta("Glossary", item, url = "https://www.segway.com.cn/", label = ["Glossary"]) for item in testcase]
        ids = [str('id' + str(id)) for id in range(len(testcase))]

        to_upsert = list(zip(ids, vectors, metas))

        my_pinecone.upsert(namespace = NamesSpaces.Glossary.value, items = to_upsert)
        print ("my_pinecone: ", my_pinecone.fetch(namespace = NamesSpaces.Glossary.value, ids = ['id0', 'id1']))

    embe = embedding_instance.get_raw_embedding("withstand voltage")
    data = my_pinecone.query_meta(namespace = NamesSpaces.Glossary.value, threshold = 0.0, vector = embe, top_k = 5)
    print ("querytest: ", data)
    print ("content str", "\n".join(item['metadata']['content'] for item in data))
