import os
import streamlit  as st
import requests

#  IMPORTANT:  This  URL  uses  the  n8n  service  name  from  docker-compose.
#  The  final  path  segment  must  match  the  production  URL  from  your  n8n  Webhook  node.
#  Override  by  setting  the  N8N_WEBHOOK_URL  environment  variable.
N8N_WEBHOOK_URL  =  os.getenv(
    "N8N_WEBHOOK_URL",
    "http://n8n-app:5678/webhook/22398436-911c-4798-a801-789a7411d5e8"
)

if not isinstance(N8N_WEBHOOK_URL, str) or not N8N_WEBHOOK_URL.strip() or not N8N_WEBHOOK_URL.startswith(("http://", "https://")):
    raise ValueError("N8N_WEBHOOK_URL must be a non-empty http(s) URL.")

st.title("Private  AI  Assistant")
st.info("Ask  a  question  about  your  documents.")

with st.form(key='query_form'):
    question  =  st.text_input("Your  Question:",  max_chars=1000)
    submit_button  =  st.form_submit_button(label='Get  Answer')

if submit_button  and question:
    with st.spinner("Searching  documents  and  generating  an  answer..."):
        try:
            payload  =  {"question":  question}
            response  =  requests.post(N8N_WEBHOOK_URL,  json=payload,  timeout=120)
            response.raise_for_status()
            result  =  response.json()

            if result  and isinstance(result,  list)  and len(result)  >  0 and 'stdout' in result[0]:
                answer  =  result[0]['stdout']
                st.success("Answer:")
                st.write(answer)
            elif isinstance(result, dict) and 'answer' in result:
                answer = result['answer']
                st.success("Answer:")
                st.write(answer)
            else:
                st.error("The assistant returned an unexpected response format. Please try again.")
                # Log detailed error for debugging if needed, but don't show all to user
                # st.json(result)

        except requests.exceptions.Timeout:
            st.error("The request timed out. The workflow may be stalled. Please try again.")
        except Exception  as e:
            st.error(f"An  unexpected  error  occurred:  {e}")
