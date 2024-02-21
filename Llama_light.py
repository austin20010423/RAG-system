from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


n_batch = 512
n_gpu_layer = -1
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="Taiwan-LLaMa-13b-1.0.Q5_K_S.gguf",
    n_batch=n_batch,
    n_gpu_layer=n_gpu_layer,
    max_tokens=20,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

template = """Question: {question}
Answer: 你是一位聊天機器人"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "用python印出hello world"
llm_chain.invoke(question)
