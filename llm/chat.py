from llama_index import GPTVectorStoreIndex
from dotenv import load_dotenv
import tiktoken


load_dotenv()

memory = {}


def estimate_tokens(txt):
    # according to OpenAI
    # https://platform.openai.com/tokenizer
    num_tokens = round(len(txt) / 4)
    return num_tokens

# https://blog.devgenius.io/counting-tokens-for-openai-gpt-3-api-59c8e0812eeb
def get_num_tokens(txt, encoding_name="gpt2"):
    enc = tiktoken.get_encoding(encoding_name)
    num_tokens = len(enc.encode(txt))
    return num_tokens

def create_answer(vector_index, txt):
    v_index = GPTVectorStoreIndex.load_from_disk(vector_index)

    prompt = f"Give me the next instruction step for article table with article number 123 based on my current action: {txt}"
    memory["current"] = txt
    # only check the memory for correctness if it's not the first action
    if memory.get("expected") is not None:
        expected_action = memory["expected"]
        # compare expected action with current action and correct mistake
        check_prompt = f"The current action is: {txt}. The expected action is: {expected_action}"\
                        + " check if they semantically align. And if so, return 'actions align'. Otherwise return 'No'"
        check_response = v_index.query(check_prompt, response_mode='compact')
        print(f"Check Reponse: {check_response}")
        if "No" in check_response.response:
            return f"Attention, you are making an error, the correct action is:\n{memory['expected']}"
    print(f"estimated tokens: {estimate_tokens(txt)}")
    print(f"calculated tokens: {get_num_tokens(txt)}\n")
    response = v_index.query(prompt, response_mode='compact')
    memory["expected"] = response
    # might be useful later to have the last action stored in the memory
    memory["last"] = memory["current"]
    #print(dir(response))
    #print(type(response.response))
    return response

if __name__ == "__main__":
    while True:
        txt = input("Your question: ")
        print(create_answer("chat/data/vector_index.json", txt))