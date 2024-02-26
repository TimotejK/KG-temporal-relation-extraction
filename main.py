import datasets.knowledge_graph_dataset
from dataLoaders import combining_data

if __name__ == '__main__':
    # ollama = Ollama(base_url='http://localhost:11434', model="openchat:7b")
    # print(ollama.invoke("Who are you?"))
    i2b2df = combining_data.read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    datasets.knowledge_graph_dataset.get_llm_responses_only(i2b2df)
    # datasets.knowledge_graph_dataset.create_knowledge_graph_dataset(i2b2df)
    pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
