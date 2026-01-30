import os

from dotenv import load_dotenv

from typing import Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase, basic_auth
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from neo4j.time import Date

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage


# 0. Graph
load_dotenv()

class State(TypedDict): # 그래프의 상태를 정의하는 클래스
    messages: Annotated[list, add_messages] # 메시지 누적
    db_outputs: list

graph_builder = StateGraph(State) # StateGraph 생성 (대화 흐름 관리)

# 1. Generate Cypher Query
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = os.getenv('NEO4J_USERNAME')
neo4j_password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(
    neo4j_uri,
    auth=basic_auth(neo4j_username, neo4j_password)
)

# 1-1. get schema
def get_node_datatype(value):
    """
        입력된 노드 Value의 데이터 타입을 반환하는 함수
    """
    if isinstance(value, str):
        return "STRING"
    elif isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "FLOAT"
    elif isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, list):
        return f"LIST[{get_node_datatype(value[0])}]" if value else "LIST"
    elif isinstance(value, Date):
        return "DATE"
    else:
        return "UNKNOWN"
    
def get_schema_dict():
    """
        Graph DB의 정보를 받아 노드 및 관계의 프로퍼티를 추출하고 스키마 딕셔너리를 반환하는 함수
    """
    with driver.session() as session:
        # 노드 프로퍼티 및 타입 추출
        node_query = """
        MATCH (n)
        WITH DISTINCT labels(n) AS node_labels, keys(n) AS property_keys, n
        UNWIND node_labels AS label
        UNWIND property_keys AS key
        RETURN label, key, n[key] AS sample_value
        """
        nodes = session.run(node_query)

        # 관계 프로퍼티 및 타입 추출
        rel_query = """
        MATCH ()-[r]->()
        WITH DISTINCT type(r) AS rel_type, keys(r) AS property_keys, r
        UNWIND property_keys AS key
        RETURN rel_type, key, r[key] AS sample_value
        """
        relationships = session.run(rel_query)

        # 관계 유형 및 방향 추출
        rel_direction_query = """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a) AS start_label, type(r) AS rel_type, labels(b) AS end_label
        ORDER BY start_label, rel_type, end_label
        """
        rel_directions = session.run(rel_direction_query)

        # 스키마 딕셔너리 생성
        schema = {"nodes": {}, "relationships": {}, "relations": []}

        for record in nodes:
            label = record["label"]
            key = record["key"]
            sample_value = record["sample_value"] # 데이터 타입을 추론하기 위한 샘플 데이터
            inferred_type = get_node_datatype(sample_value)
            if label not in schema["nodes"]:
                schema["nodes"][label] = {}
            schema["nodes"][label][key] = inferred_type

        for record in relationships:
            rel_type = record["rel_type"]
            key = record["key"]
            sample_value = record["sample_value"] # 데이터 타입을 추론하기 위한 샘플 데이터
            inferred_type = get_node_datatype(sample_value)
            if rel_type not in schema["relationships"]:
                schema["relationships"][rel_type] = {}
            schema["relationships"][rel_type][key] = inferred_type

        for record in rel_directions:
            start_label = record["start_label"][0]
            rel_type = record["rel_type"]
            end_label = record["end_label"][0]
            schema["relations"].append(f"(:{start_label})-[:{rel_type}]->(:{end_label})")

        return schema

def get_schema_str(schema):
    """
        스키마 딕셔너리를 LLM에 제공하기 위해 원하는 형태로 formatting 하는 함수 
    """
    result = []

    # 노드 프로퍼티 출력
    result.append("Node properties:")
    for label, properties in schema["nodes"].items():
        props = ", ".join(f"{k}: {v}" for k, v in properties.items())
        result.append(f"{label} {{{props}}}")

    # 관계 프로퍼티 출력
    result.append("Relationship properties:")
    for rel_type, properties in schema["relationships"].items():
        props = ", ".join(f"{k}: {v}" for k, v in properties.items())
        result.append(f"{rel_type} {{{props}}}")

    # 관계 프로퍼티 출력
    result.append("The relationships:")
    for relation in schema["relations"]:
        result.append(relation)

    return "\n".join(result)

schema = get_schema_str(get_schema_dict())
# print(schema)

# 1-2. llm for generating cypher
google_api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

## LLM INPUT / QUERY 예시 제공
fewshot_examples = [
    "USER INPUT: 'Toy Story에 어떤 배우들이 출연하나요?' QUERY: MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'Toy Story' RETURN a.name",
    "USER INPUT: 'Toy Story의 평균 평점은 몇점인가요?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Toy Story' RETURN AVG(r.rating)",
]

GENERATE_SYSTEM_TEMPLATE = """Given an input question, convert it to a Cypher query. No pre-amble.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

GENERATE_USER_TEMPLATE = """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""

def generate_cypher(state: State):
    generate_cypher_msgs = [
        ("system", GENERATE_SYSTEM_TEMPLATE),
        ("user", GENERATE_USER_TEMPLATE),
    ]
    text2cypher_prompt = ChatPromptTemplate.from_messages(generate_cypher_msgs)

    response = llm.invoke(
        text2cypher_prompt.format_messages(
            question=state["messages"], schema=schema, fewshot_examples=fewshot_examples
        )
    )
    outputs = []
    outputs.append(
        AIMessage(
            content=response.content,
        )
    )
    return {"messages": outputs}

graph_builder.add_node("generate_cypher", generate_cypher)

# 2. Excute Query
class ExcuteCypherNode:
    """
        A node that runs the query generated in the last AIMessage.
        마지막 AIMessage에서 생성된 Cypher를 실행하는 노드
    """
    def __init__(self) -> None:
        self.driver = driver

    def __call__(self, inputs: dict):
        print("###### EXCUTE CYPHER ######")
        if messages := inputs.get("messages", []):
            message = messages[-1].content # 마지막 message
        else:
            raise ValueError("No message found in input")
        outputs = []

        print("실행 쿼리문", message)

        neo4j_database = os.getenv('NEO4J_DATABASE')
        try:
            with driver.session(database=neo4j_database) as session:
                database_output = session.execute_read(
                    lambda tx: tx.run(message).data())
        except Exception as e:
            database_output = str(e)

        outputs.append(
            database_output
        )
        return {"db_outputs": outputs}
    
excute_cypher_node = ExcuteCypherNode()
graph_builder.add_node("excute_cypher", excute_cypher_node)

# 2-1. Router Edge
def route_correction(state: State):
    """
        A edge that route to the "correct_cypher" node if the last generated Cypher query was not executed properly.
        마지막으로 생성된 Cypher 쿼리가 제대로 실행되지 않은 경우 "correct_cypher" 노드로 라우팅되는 엣지
    """
    if db_outputs := state.get("db_outputs", []):
        db_result = db_outputs[-1]
    else:
        raise ValueError("No DB result found")
    
    print("###### ROUTE QUERY CORRECTION ######")
    print("DB 조회 결과", db_result)
    if type(db_result) == list and len(db_result) > 0: # 실행 성공 조건 : DB 조회 결과가 길이가 1이상인 리스트
        print("!정상 조회 완료!")
        return "answer"
    
    print("!정상 조회 실패!")
    return "correct_cypher"

## 조건부 엣지 연결
graph_builder.add_conditional_edges(
    "excute_cypher", # 시작 노드
    route_correction,
    {"correct_cypher": "correct_cypher", "answer": "answer"},
)

# 3. Correct Query
CORRECT_CYPHER_SYSTEM_TEMPLATE = """You are a Cypher expert reviewing a statement written by a junior developer.
You need to correct the Cypher statement based on the provided errors. No pre-amble."
If the error is empty, return "RETURN "No search results found." AS result;".
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

CORRECT_CYPHER_USER_TEMPLATE = """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """

def correct_cypher(state: State):
    if messages := state.get("db_outputs", []):
        db_result = messages[-1]
    else:
        raise ValueError("No DB result found")
    
    if messages := state.get("messages", []):
        cypher = messages[-1].content
    else:
        raise ValueError("No Cypher found")
    
    print("###### CORRECT CYPHER ######")
    print("수정 전 쿼리문", cypher)
    print("DB 조회 결과", db_result)

    correct_cypher_messages = [
        ("system", CORRECT_CYPHER_SYSTEM_TEMPLATE),
        ("user", CORRECT_CYPHER_USER_TEMPLATE),
    ]
    correct_cypher_prompt = ChatPromptTemplate.from_messages(correct_cypher_messages)

    response = llm.invoke(
        correct_cypher_prompt.format_messages(
            question=state["messages"], schema=schema, cypher=cypher, errors=db_result
        )
    )

    outputs = []
    outputs.append(
        AIMessage(
            content=response.content,
        )
    )

    print("수정 후 쿼리문", response.content)
    return {"messages": outputs}

graph_builder.add_node("correct_cypher", correct_cypher)

# 4. Answer based DB Results
FINAL_ANSWER_SYSTEM_TEMPLATE = """
You are a highly intelligent assistant trained to provide concise and accurate answers.
You will be given a context that has been retrieved from a Neo4j database using a specific Cypher query.
Your task is to analyze the context and answer the user’s question based on the information provided in the context.
If the context lacks sufficient information, inform the user and suggest what additional details are needed.

Focus solely on the context provided from the Neo4j database to form your response.
Avoid making assumptions or using external knowledge unless explicitly stated in the context.
Ensure the final answer is clear, relevant, and directly addresses the user’s question.
If the question is ambiguous, ask clarifying questions to ensure accuracy before proceeding.
ANSWER IN KOREAN.
"""

FINAL_ANSWER_USER_TEMPLATE = """
Based on this context retrieved from a Neo4j database using the following Cypher query:
`{cypher_query}`

The context is:
{context}

Answer the following question:
<question>
{question}
</question>

Please provide your answer based on the context above, explaining your reasoning.
If clarification or additional information is needed, explain why and specify what is required.
"""

def answer(state: State):
    if messages := state.get("db_outputs", []):
        db_result = messages[-1]
    else:
        raise ValueError("No DB result found")
    
    if messages := state.get("messages", []):
        cypher = messages[-1].content
    else:
        raise ValueError("No Cypher found")
    
    final_answer_msgs = [
        ("system", FINAL_ANSWER_SYSTEM_TEMPLATE),
        ("user", FINAL_ANSWER_USER_TEMPLATE),
    ]
    final_answer_prompt = ChatPromptTemplate.from_messages(final_answer_msgs)

    response = llm.invoke(
        final_answer_prompt.format_messages(
            cypher_query=cypher, context=db_result, question=state["messages"]
        )
    )

    outputs = []
    outputs.append(
        AIMessage(
            content=response.content,
        )
    )

    print("###### ANSWER BASED DB RESULT ######")
    print("최종 답변", response.content)

    return {"messages": outputs}

graph_builder.add_node("answer", answer)

# 5. Compile Graph
## 엣지 연결
graph_builder.add_edge(START, "generate_cypher")
graph_builder.add_edge("generate_cypher", "excute_cypher")
graph_builder.add_edge("correct_cypher", "excute_cypher")
graph_builder.add_edge("answer", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}): # graph 노드 호출 결과 받아옴
        for value in event.values():
            print(value, "\n")

while True:
    try:
        user_input = input("🧑‍💻 User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Error: {e}")
        break