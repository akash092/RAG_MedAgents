from prompt_generator import get_question_analysis_prompt
RAG_AGENT_DOMAIN = "Hypertension"

class RAGAgent():
    def __init__(self, retriever, handler):
        self.handler = handler
        self.retriever = retriever

    def chat(self, question: str, top_k: int = 5):
        # 1. Retrieve relevant docs
        retrieved_docs = self.retriever.get_relevant_documents(question, top_k=top_k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. Build User prompt
        _, prompt_get_question_analysis = get_question_analysis_prompt(question, RAG_AGENT_DOMAIN)

        # 2.1 Build system prompt
        question_analyzer = f"You are a medical expert in the domain of {RAG_AGENT_DOMAIN}. "\
            f"Using the following retrieved context, you will scrutinize and diagnose the symptoms presented by patients in specific medical scenarios. Context:{context}"

        # 3. Generate with the LLM
        answer = self.handler.get_output_multiagent(user_input=prompt_get_question_analysis, max_tokens=300, system_role=question_analyzer)

        # 4. Return
        return answer