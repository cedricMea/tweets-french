from transformers import pipeline
import urllib3
import os 

if __name__=="__main__":
    # Avoid Milliman SSL  
    os.environ["CURL_CA_BUNDLE"] = ""
    # disable warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    nlp_qa = pipeline("question-answering", model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased') 
    cont_string = "The mortality is reported to be around 3.4% and r0 rate is 2% "
    ques_string = "What is r0 rate value ?"
    ques_string_1 = "What is r 0 ?"
    ques_string_2 = "What is r0 ?"


    res = nlp_qa(context=cont_string, question=ques_string)

    print(res)
    print("#"*60)
    print(nlp_qa(context=cont_string, question=ques_string_1))
    print("#"*60)
    print(nlp_qa(context=cont_string, question=ques_string_2))

