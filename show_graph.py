"""打印 LangGraph 的流程图（Mermaid 格式）"""
from langgraph_demo import graph

mermaid = graph.get_graph().draw_mermaid()
print(mermaid)
