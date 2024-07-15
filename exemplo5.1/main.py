from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


llm = Ollama(
    model="llama2",
    num_gpu=0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

def gerar_insights_sobre_filmes(pergunta):
    prompt = f"Responda a seguinte pergunta sobre filmes: {pergunta}\n"
    prompt += "Por favor, forneça um resumo detalhado e quaisquer informações relevantes."

    insights = llm.invoke(prompt)
    return insights

def responder_pergunta(pergunta):
    resposta = gerar_insights_sobre_filmes(pergunta)
    return resposta

def main():
    pergunta = input("Sobre qual filme você deseja saber informações? ")
    resposta = responder_pergunta(pergunta)
    print(f"Resposta: {resposta}")

if __name__ == "__main__":
    main()
