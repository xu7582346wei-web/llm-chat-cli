from langgraph_demo import graph
png = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)
print("已生成 graph.png")
