import streamlit  as st
import requests

#  IMPORTANT:  This  URL  uses  the  n8n  service  name  from  docker-compose.
#  The  final  path  segment  must  match  the  production  URL  from  your  n8n  Webhook  node.
N8N_WEBHOOK_URL  =  "http://n8n-app:5678/webhook/22398436-911c-4798-a801-789a7411d5e8"

st.title("Private  AI  Assistant")
st.info("Ask  a  question  about  your  documents.")

with st.form(key='query_form'):
    question  =  st.text_input("Your  Question:")
    submit_button  =  st.form_submit_button(label='Get  Answer')

if submit_button  and question:
    with st.spinner("Searching  documents  and  generating  an  answer..."):
        try:
            payload  =  {"question":  question}
            # Added a timeout of 60 seconds to avoid hanging requests
            response  =  requests.post(N8N_WEBHOOK_URL,  json=payload, timeout=60)
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
            st.error("The request timed out. The model might be busy or the documents are too large.")
        except requests.exceptions.RequestException:
            st.error("Could not connect to the assistant service. Please ensure the backend is running.")
        except Exception:
            st.error("An unexpected error occurred. Please contact support.")
