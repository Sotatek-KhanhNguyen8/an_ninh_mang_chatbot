import os
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext, load_index_from_storage
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import streamlit as st

os.environ['OPENAI_API_KEY'] = ''
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1024)
Settings.context_window = 15000
import csv
import sqlite3

# Tên của tập tin CSV
csv_file = 'luat_mang.csv'

# Kết nối đến cơ sở dữ liệu SQLite hoặc tạo một cơ sở dữ liệu mới
# conn = sqlite3.connect('database.db')
# cursor = conn.cursor()
#
# # Tạo bảng trong cơ sở dữ liệu
# cursor.execute('''CREATE TABLE IF NOT EXISTS du_lieu (
#                     name TEXT,
#                     chapter TEXT,
#                     title TEXT,
#                     content TEXT
#                     )''')
#
# # Đọc dữ liệu từ tập tin CSV và chèn vào bảng
# with open(csv_file, 'r', encoding='utf-8') as file:
#     csv_reader = csv.reader(file)
#     next(csv_reader)  # Bỏ qua dòng tiêu đề nếu có
#     for row in csv_reader:
#         cursor.execute('''INSERT INTO du_lieu (name, chapter, title, content)
#                         VALUES (?,?,?,?)''', row)
# conn.commit()
# conn.close()
with open('luat_raw.txt', 'r', encoding='utf-8') as f:
    # Đọc nội dung của file
    content = f.read()
from llama_index.core import Document
doc = Document(text = content)
documents=[]
documents.append(doc)
# Lưu thay đổi và đóng kết nối
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

nodes = splitter.get_nodes_from_documents(documents)
# index = VectorStoreIndex(nodes)
# index.storage_context.persist(persist_dir="db")
storage_context = StorageContext.from_defaults(persist_dir="db")

# load index
index = load_index_from_storage(storage_context)
llm = OpenAI(model="gpt-3.5-turbo", temperature =0)
few_shot_examples="""
"Query": "Nội dung của điều 7 là"
"Answer": SELECT * FROM du_lieu WHERE name LIKE \'%7%\';

"Query": "Nội dung của điều 18 là"
"Answer": SELECT * FROM du_lieu WHERE name LIKE \'%18%\';

"Query": "Hệ thống thông tin quan trọng về an ninh quốc gia là gì?"
"Answer": SELECT * FROM du_lieu WHERE title LIKE \'%Hệ thống thông tin quan trọng về an ninh quốc gia%\';

"Query": "chương 2 có bao nhiêu điều"
"Answer": SELECT count(*) FROM du_lieu WHERE chapter LIKE \'%2%\';

"Query": "Có bao nhiêu điều luật về cách Phòng ngừa và xử lý hành vi xâm phạm an ninh mạng"
"Answer": SELECT COUNT(*) FROM du_lieu WHERE chapter LIKE \'%Phòng ngừa và xử lý hành vi xâm phạm an ninh mạng%\';


"""
# write prompt template with functions
from llama_index.core import PromptTemplate
qa_prompt_tmpl_str = """
Bạn là một chuyên gia SQLite. Đưa ra một câu hỏi đầu vào, hãy tạo một truy vấn SQLite đúng về mặt cú pháp để chạy. Sử dụng 'LIKE' thay vì '='.
Dưới đây là thông tin các cột của bảng du_lieu trong cơ sở dữ liệu:
name, chapter, title, content
Chú ý tham khảo thêm thông tin dưới đây để xác định đúng trường nếu người dùng nhập sai:
---------------------
{context_str}
---------------------
Một số ví dụ được đưa ra dưới đây:

{few_shot_examples}

Query: {query_str}
Answer:
"""
additional_args = {"context_str": "{context_str}", "query_str": "{query_str}", "few_shot_examples": few_shot_examples}
qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str,
    # format =additional_args,
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl.format(**additional_args))
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# build index

# configure retriever
retriever = index.as_retriever(similarity_top_k=1)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
#     # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
# )
query_engine_sql = RetrieverQueryEngine.from_args(
    retriever, response_mode='compact'
)
query_engine_sql.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)
prompts_dict = query_engine_sql.get_prompts()
# display_prompt_dict(prompts_dict)
# query
response = query_engine_sql.query("nội dung của điều 23 là")
# print(response)
def run_sql(response):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Thực hiện truy vấn
    cursor.execute(response.response.lower())
    # Lấy kết quả truy vấn
    rows = cursor.fetchall()
    # Đóng kết nối
    conn.close()
    # print(str(rows))
    return str(rows)
from llama_index.core.tools import FunctionTool

def retriever_syns(query: str) -> str:
    """đưa chi tiết câu hỏi vào đây"""
    print('switch to 2')
    # response = query_engine_sql.query(query)
    # print(response.response)
    # result = run_sql(response)
    # print(result)
    context_str = '{context_str}'
    query_str = '{query_str}'
    final_prompt_tmpl_str = f"""
    Đưa ra câu hỏi của người dùng sau. Nếu không có thông tin hãy trả lời là không có thông tin
    Context: {context_str}
    Query: {query_str}
    Answer:
    """
    # additional_args = {"context_str": "{context_str}", "query_str": "{query_str}", "few_shot_examples": few_shot_examples}
    final_prompt_tmpl = PromptTemplate(
        final_prompt_tmpl_str,
        # format =additional_args,
    )
    retriever = index.as_retriever()

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()
    query_engine_retriever = RetrieverQueryEngine.from_args(
    retriever, response_mode='compact'
    )
    query_engine_retriever.update_prompts(
    {"response_synthesizer:text_qa_template": final_prompt_tmpl}
    )
    # prompts_dict = query_engine_retriever.get_prompts()
    # display_prompt_dict(prompts_dict)
    # prompts_dict = query_engine.get_prompts()
    # display_prompt_dict(prompts_dict)
    # query
    ans = query_engine_retriever.query(query)

    return ans.response


retriever_tool = FunctionTool.from_defaults(fn=retriever_syns, return_direct=True)
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer

qa_prompt = PromptTemplate(
    "Đưa ra câu hỏi của người dùng sau, sử dụng thông tin từ truy vấn SQL để trả lời câu hỏi của người dùng. Nếu không có thông tin hãy in ra dòng chữ 'Không có thông tin.'"
    "SQL Query: {sql_query}"
    "SQL Result: {sql_result}"
    "---------------------\n"
    "Sử dụng thông tin từ truy vấn SQL để trả lời câu hỏi của người dùng\n"
    "Query: {query_str}\n"
    "Answer: "
)


from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
def Chat_sql_and_retriever(query: str) -> str:
    """Hữu ích khi trả lời các câu hỏi về các điều luật an ninh mạng"""
    try:

        response = query_engine_sql.query(query)
        # print(response.response)
        result = run_sql(response)
        # print(result)
        context_str = '{context_str}'
        query_str = '{query_str}'
        final_prompt_tmpl_str = f"""
        Đưa ra câu hỏi của người dùng sau, kết hợp giữa thông tin từ truy vấn SQL và thông tin Data để trả lời câu hỏi của người dùng. Nếu không có thông tin từ SQL, hãy lấy thông tin từ Data. Nếu không có thông tin từ cả hai hãy in ra dòng chữ 'Không có thông tin'.
        SQL Query: {response.response}
        SQL Result: {result}
        Data: {context_str}
        Query: {query_str}

        Answer:
        """
        # additional_args = {"context_str": "{context_str}", "query_str": "{query_str}", "few_shot_examples": few_shot_examples}
        final_prompt_tmpl = PromptTemplate(
            final_prompt_tmpl_str,
            # format =additional_args,
        )
        retriever = index.as_retriever(similarity_top_k=1)

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer()
        query_engine_retriever = RetrieverQueryEngine.from_args(
        retriever, response_mode='compact'
        )
        query_engine_retriever.update_prompts(
        {"response_synthesizer:text_qa_template": final_prompt_tmpl}
        )
        # prompts_dict = query_engine_retriever.get_prompts()
        # display_prompt_dict(prompts_dict)
        # prompts_dict = query_engine.get_prompts()
        # display_prompt_dict(prompts_dict)
        # query
        ans = query_engine_retriever.query(query)
        # query_engine_retriever.query()
        return ans.response
    except:
        return 'Không có thông tin1'


multiply_tool = FunctionTool.from_defaults(fn=Chat_sql_and_retriever,return_direct=True)
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer

qa_prompt = PromptTemplate(
    "Đưa ra câu hỏi của người dùng sau, sử dụng thông tin từ truy vấn SQL để trả lời câu hỏi của người dùng. Cho biết các cột trong bảng du_lieu là: name, chapter, title, content. name cho biết điều số bao nhiêu, chapter cho biết điều đó thuộc chương nào, title cho biết tiêu đề, content là nội dung chi tiết Nếu không có thông tin hãy in ra dòng chữ 'Không có thông tin.'"
    "SQL Query: {sql_query}"
    "SQL Result: {sql_result}"
    "---------------------\n"
    "Sử dụng thông tin từ truy vấn SQL để trả lời câu hỏi của người dùng\n"
    "Query: {query_str}\n"
    "Answer: "
)


class RAGStringQueryEngine(CustomQueryEngine):
    """SQL Query Engine."""

    # retriever: BaseRetriever
    # response_synthesizer: BaseSynthesizer
    llm: llm
    qa_prompt: PromptTemplate


    def custom_query(self, query_str: str, sql_query: str, sql_result: str):
        # nodes = self.retriever.retrieve(query_str)

        # context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            qa_prompt.format(query_str=query_str, sql_query=sql_query, sql_result=sql_result)
        )

        return response
def product_tool(query: str) -> str:
    """Lấy toàn bộ câu hỏi của người dùng truyền vào đây, không được bỏ sót một từ nào"""
    try:
        response = query_engine_sql.query(query)
        # print(response.response)
        sql_query = response.response
        result = run_sql(response)
        # print(result)
        query_engine = RAGStringQueryEngine(
        # retriever=retriever,
        # response_synthesizer=synthesizer,
        llm=llm,
        qa_prompt=qa_prompt,
        # sql_query=response.response,
        # sql_result=result
        )
        ans = query_engine.custom_query(query, sql_query=sql_query, sql_result=result)
        # prompts_dict = query_engine_retriever.get_prompts()
        # display_prompt_dict(prompts_dict)
        # prompts_dict = query_engine.get_prompts()
        # display_prompt_dict(prompts_dict)
        # query
        # ans = query_engine_retriever.query(query)
        # query_engine_retriever.query()
        answer = ans.text
    except:
        answer = 'Không có thông tin'
    if 'hông có thông tin' in answer:
        answer = retriever_syns(query)
    return answer
        # return 'Không có thông tin1'


product_tool = FunctionTool.from_defaults(fn=product_tool,return_direct=True)
from llama_index.agent.openai import OpenAIAgent, OpenAIAgentWorker
from llama_index.llms.openai import OpenAI
agent = OpenAIAgent.from_tools(
    [product_tool],
    llm=llm,
    # verbose=True,
)
# print('start')
# while True:
#     text = input()
#     print(agent.chat(text).response)

# # answer = query_engine_retriever.query('30/4 được nghỉ có lương không')
app = FastAPI()
class TextRequest(BaseModel):
    question: str

@app.post("/chat/")
def chat(request: TextRequest):
    response = agent.chat(request.question).response
    return response
if __name__ == "__main__":
    uvicorn.run(app, port=5006, host='0.0.0.0')
