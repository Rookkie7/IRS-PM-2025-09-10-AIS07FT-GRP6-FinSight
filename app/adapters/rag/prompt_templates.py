def build_finance_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        [f"[{i+1}] {c.get('title','(untitled)')}\n{c.get('text','')[:800]}"
         for i, c in enumerate(contexts)]
    )
    return f"""You are a helpful financial analysis assistant.
            Use ONLY the following context to answer the user's question. If the answer is unknown, say so.
            
            Context:
            {context_text}
            
            Question: {question}
            
            Instructions:
            - Cite sources as [1], [2], ... based on the context order when relevant.
            - Be concise and factual.
            Answer:"""