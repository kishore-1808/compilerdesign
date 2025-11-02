from transformers import pipeline
import os
import json
import re
import io
import sys
import tiktoken
import pydot
import streamlit as st
from pycparser import c_parser, c_lexer


# ------------------------------------------------------------
# Redirect Hugging Face cache to D drive
# ------------------------------------------------------------
os.environ["HF_HOME"] = "D:\\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface_cache\\transformers"

# ------------------------------------------------------------
# PHASE 1: Lexical Analysis
# ------------------------------------------------------------
def lexical_analysis(code):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(code))

    def _error(msg, line, column):
        print(f"Lexer Error → line {line}, column {column}: {msg}")

    def _on_lbrace(): pass
    def _on_rbrace(): pass
    def _type_lookup(typ): return False

    lexer = c_lexer.CLexer(_error, _on_lbrace, _on_rbrace, _type_lookup)
    lexer.build()
    lexer.input(code)

    tokens = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens.append((tok.type, tok.value))

    token_output = [f"{t[0]} → {t[1]}" for t in tokens]
    return tokens, token_output, token_count

# ------------------------------------------------------------
# PHASE 2: Parser Phase
# ------------------------------------------------------------
def parser_phase(code):
    parser = c_parser.CParser()
    try:
        ast_tree = parser.parse(code)
        ast_json = ast_to_json(ast_tree)
        json_output = json.dumps(ast_json, indent=4)
        return ast_tree, "✅ Code syntax is valid.", json_output
    except Exception as e:
        return None, f"❌ Parser Error: {e}", None

def ast_to_json(node):
    if node is None:
        return None
    result = {"type": type(node).__name__}
    children = []
    for name, child in node.children():
        children.append({"name": name, "child": ast_to_json(child)})
    if children:
        result["children"] = children
    return result

# ------------------------------------------------------------
# PHASE 3: AST Visualization
# ------------------------------------------------------------
def build_pydot(node, graph, parent=None):
    label = type(node).__name__
    current = pydot.Node(
        str(id(node)),
        label=label,
        shape="box",
        style="filled",
        fillcolor="lightblue",
    )
    graph.add_node(current)
    if parent:
        graph.add_edge(pydot.Edge(str(id(parent)), str(id(node))))
    for _, child in node.children():
        build_pydot(child, graph, node)

def ast_visualization(ast_tree):
    graph = pydot.Dot("C_AST", graph_type="digraph", rankdir="TB")
    build_pydot(ast_tree, graph)
    img_path = "c_ast_tree.png"
    graph.write_png(img_path)
    return img_path

# ------------------------------------------------------------
# PHASE 4: LLM Feedback
# ------------------------------------------------------------
def get_llm_feedback(code):
    llm = pipeline("text-generation", model="sshleifer/tiny-gpt2")

    prompt = (
        "Analyze the following C code and respond strictly in JSON format:\n"
        "{\n"
        '  "correctness": "Correct" or "Incorrect",\n'
        '  "suggestions": ["suggestion1", "suggestion2"]\n'
        "}\n\n"
        f"Code:\n{code}\n\nResponse:"
    )

    result = llm(prompt, max_new_tokens=150)[0]["generated_text"]
    feedback_text = result.split("Response:", 1)[-1].strip()

    cleaned = re.sub(r"[^a-zA-Z0-9\s.,;:(){}'\"-]", "", feedback_text)
    cleaned_lower = cleaned.lower()

    if any(word in cleaned_lower for word in ["error", "syntax", "invalid", "fail"]):
        correctness = "Incorrect"
    elif any(word in cleaned_lower for word in ["success", "valid", "correct"]):
        correctness = "Correct"
    else:
        correctness = "Unknown"

    suggestions = []
    if "/0" in code or "b = 0" in code:
        suggestions.append("Avoid dividing by zero; validate divisor before division.")
    if "printf" not in code:
        suggestions.append("Consider adding printf statements for debugging.")
    if "return" not in code:
        suggestions.append("Ensure main function returns a value.")
    if not suggestions:
        suggestions.append("Code structure seems valid. No immediate issues detected.")

    feedback_json = {
        "correctness": correctness,
        "suggestions": suggestions
    }
    return feedback_json

# ------------------------------------------------------------
# STREAMLIT UI (STYLED)
# ------------------------------------------------------------
def launch_streamlit_ui():
    st.set_page_config(page_title="AI C Compiler", layout="wide")

    # Custom CSS Styling
    st.markdown("""
        <style>
            .main {
                background-color: #0e1117;
                color: #ffffff;
            }
            .stTextArea>div>textarea {
                background-color: #1e1e1e !important;
                color: #dcdcdc !important;
                border-radius: 10px !important;
            }
            .stTextInput>div>div>input {
                background-color: #1e1e1e !important;
                color: white !important;
            }
            .section-title {
                color: #61dafb;
                font-weight: 700;
                font-size: 22px;
                margin-top: 20px;
            }
            .output-card {
                background-color: #1e1e1e;
                padding: 20px;
                border-radius: 12px;
                margin-top: 10px;
                box-shadow: 0 0 10px #61dafb33;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Adjust compiler configurations below.")
        st.markdown("---")
        model_choice = st.selectbox("Select Feedback Model", ["sshleifer/tiny-gpt2", "distilgpt2", "gpt2"])
        st.write("📁 Cache Path: D:\\huggingface_cache")
        st.markdown("---")
        st.write("👨‍💻 Developed by *Your Name*")

    # Header
    st.title("🧠 AI-Enhanced C Compiler Simulation")
    st.markdown("### Lexical → Parser → AST → LLM Feedback")
    st.markdown("---")

    # Input Section
    st.markdown("<div class='section-title'>🧮 Enter Your C Code</div>", unsafe_allow_html=True)
    code = st.text_area("C Code", height=250, value="""#include<stdio.h>
int main() {
    int a = 10, b = 20;
    int sum = a + b;
    printf("%d", sum);
    return 0;
}""")

    if st.button("🚀 Run Compiler"):
        if not code.strip():
            st.warning("⚠️ Please enter C code first.")
        else:
            with st.spinner("Compiling and analyzing code..."):
                tokens, token_output, token_count = lexical_analysis(code)
                ast_tree, parser_status, ast_json = parser_phase(code)
                img_path = ast_visualization(ast_tree) if ast_tree else None
                feedback = get_llm_feedback(code)

            # Phase 1 - Lexical Analysis
            st.markdown("<div class='section-title'>🔍 Lexical Analysis</div>", unsafe_allow_html=True)
            with st.expander(f"📘 Tokens ({token_count} total)", expanded=True):
                st.text_area("Tokens", "\n".join(token_output), height=200)

            # Phase 2 - Parser Phase
            st.markdown("<div class='section-title'>📚 Parser Output</div>", unsafe_allow_html=True)
            st.info(parser_status)
            if ast_json:
                with st.expander("🧩 AST JSON Structure", expanded=False):
                    st.json(json.loads(ast_json))

            # Phase 3 - AST Visualization
            st.markdown("<div class='section-title'>🌳 AST Visualization</div>", unsafe_allow_html=True)
            if img_path and os.path.exists(img_path):
                st.image(img_path, caption="Abstract Syntax Tree", use_container_width=True)
            else:
                st.warning("AST not generated due to syntax errors.")

            # Phase 4 - LLM Feedback
            st.markdown("<div class='section-title'>🤖 AI Feedback</div>", unsafe_allow_html=True)
            with st.expander("View AI Analysis", expanded=True):
                st.json(feedback)
    else:
        st.info("👆 Paste your C code and click 'Run Compiler' to begin.")

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    launch_streamlit_ui()
