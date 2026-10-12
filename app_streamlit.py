"""Streamlit Web UI for LLM Agent"""
import streamlit as st
import agent as agent_module
from agent import Agent, VectorStore, load_documents
from config import KNOWLEDGE_DIR, MODEL, EMBED_MODEL

st.set_page_config(
    page_title="LLM Agent",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource(show_spinner="加载知识库中...")
def get_vector_store():
    """全局只加载一次向量库。"""
    store = VectorStore()
    chunks = load_documents(KNOWLEDGE_DIR)
    if chunks:
        store.add(chunks)
    agent_module.STORE = store
    return len(chunks)


def init_session():
    if "agent" not in st.session_state:
        st.session_state.agent = Agent()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_tools" not in st.session_state:
        st.session_state.show_tools = True
    if "current_events" not in st.session_state:
        st.session_state.current_events = []


def on_tool_event(kind, data):
    if kind == "tool_call":
        st.session_state.current_events.append({
            "kind": "tool_call",
            "name": data["name"],
            "args": data["args"],
            "result": data["result"],
        })
    elif kind == "plan":
        st.session_state.current_events.append({
            "kind": "plan",
            "plan": data["plan"],
        })


def render_sidebar(n_chunks):
    with st.sidebar:
        st.title("⚙️ 设置")
        st.markdown(f"**模型**：`{MODEL}`")
        st.markdown(f"**Embedding**：`{EMBED_MODEL}`")
        st.markdown(f"**知识库**：{n_chunks} 个片段")
        st.divider()

        st.session_state.show_tools = st.toggle(
            "显示工具调用", value=st.session_state.show_tools
        )

        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.agent.clear()
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.caption(f"历史消息数：{len(st.session_state.history)}")


def render_events(events):
    if not events:
        return
    with st.expander(f"🔧 执行过程（{len(events)} 步）", expanded=False):
        for i, e in enumerate(events, 1):
            if e["kind"] == "plan":
                st.markdown(f"**{i}. 📋 任务规划**")
                for item in e["plan"]:
                    st.markdown(f"- Step {item.get('step', '?')}: {item.get('task', '')}")
            elif e["kind"] == "tool_call":
                st.markdown(f"**{i}. 🔧 `{e['name']}`**")
                st.code(
                    f"参数：{e['args']}\n结果：{e['result']}",
                    language="text",
                )


def render_history():
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            if st.session_state.show_tools:
                render_events(turn.get("events", []))
            st.write(turn["assistant"])


def handle_input(prompt):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st.session_state.current_events = []
        with st.spinner("思考中..."):
            try:
                answer = st.session_state.agent.chat(
                    prompt,
                    verbose=False,
                    on_event=on_tool_event,
                )
            except Exception as e:
                answer = f"❌ 出错：{e}"

        events = list(st.session_state.current_events)
        if st.session_state.show_tools:
            render_events(events)
        st.write(answer)

    st.session_state.history.append({
        "user": prompt,
        "assistant": answer,
        "events": events,
    })


def main():
    init_session()
    n_chunks = get_vector_store()
    render_sidebar(n_chunks)

    st.title("🤖 LLM Agent")
    st.caption("工具调用 · 长期记忆 · 任务规划 · RAG 知识库")

    render_history()

    if prompt := st.chat_input("输入消息..."):
        handle_input(prompt)


if __name__ == "__main__":
    main()
