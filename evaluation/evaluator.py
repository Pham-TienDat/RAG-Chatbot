import numpy as np
from dotenv import load_dotenv
from trulens.apps.langchain import TruChain
from trulens.core import Metric, Selector, TruSession
from trulens.providers.litellm import LiteLLM

from core_rag import build_rag_chain, get_vectorstore

load_dotenv()

provider = LiteLLM(model_engine="gemini/gemini-3.8-flash")

# 1. Đánh giá Groundedness
f_groundedness = Metric(
    implementation=(
        provider.groundedness_measure_with_cot_reasons_consider_answerability
    ),
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
        "question": Selector.select_record_input(),
    },
)

# 2. Đánh giá Answer Relevance
f_answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)

# 3. Đánh giá Context Relevance
f_context_relevance = Metric(
    implementation=provider.context_relevance_with_cot_reasons,
    name="Context Relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),
    },
    agg=np.mean,
)


def main():
    session = TruSession()
    session.reset_database()
    rag = build_rag_chain(get_vectorstore(), k=2, temperature=0.2, max_tokens=512)
    tru_chain = TruChain(
        rag,
        app_name="Chat",
        app_version="v1",
        feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
    )
    with tru_chain as recording:
        rag.invoke({"input": "Cấu tạo của vi khuẩn"})
    results = recording[0].retrieve_feedback_results(timeout=300)
    print(results)
    print(session.get_leaderboard())


if __name__ == "__main__":
    main()
