https://www.youtube.com/watch?v=XNmFIkViEBU

STEPS:

1. create an account on HuggingFace and get your API KEY
2. create a .env file and insert:
   HUGGINGFACEHUB_API_TOKEN=**PUT YOUR API KEY HERE**
3. look for llama2 7Gb chat on HuggingFace (for example https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) or for the 13Gb

ATTENTION
Note: Use of this model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the website and accept our License before requesting access here.
It is necessary to request access at Meta website: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
BEWARE: the account (the email) used on both HugginFace and Meta must be the same otherwise it won't work

you will receive by email the response with the link to access the granted version (on HuggingFace); when accessing there is on a message saying:
Gated model
You have been granted access to this model

4. one option is to have the download of the model occurring via "code" and it is necessary to provide the HuggingFace Token to do that; please use the "use_auth_token" parameter:

model_id="meta-llama/Llama-2-7b-chat-hf"
HF_AUTH_TOKEN=os.getenv('HF_AUTH_TOKEN')

model_config = transformers.AutoConfig.from_pretrained(
model_id,
use_auth_token=HF_AUTH_TOKEN
)

model = transformers.AutoModelForCausalLM.from_pretrained(
model_id,
trust_remote_code=True,
config=model_config,
quantization_config=bnb_config,
device_map='auto',
use_auth_token=HF_AUTH_TOKEN
)

in this case it will be downloaded in the folder: C:\Users\YOURUSERNAME\.cache\huggingface\hub
BEWARE: the files and in a BLOB and it seems that it is not possible to handle them directy

HOW TO RUN LOCALLY: https://agi-sphere.com/llama-2/
Depending on having CPU or GPU it is possible to use different FILES:
- CPU: https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin

local models on HuggingFace are located here: https://huggingface.co/localmodels
