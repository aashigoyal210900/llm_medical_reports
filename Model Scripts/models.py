
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import spacy
import textwrap
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from transformers import pipeline
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
from summa import summarizer
from rouge import Rouge


df = pd.read_csv('mtsamples - annotated.csv')
df = df.drop(columns=['Unnamed: 0', 'medical_specialty','sample_name','keywords'])
df.isnull().sum()
df = df.dropna().reset_index()
transcription = list(df['transcription'])
reference = list(df['summary'])


punctuation += '\n' 
stopwords = list(STOP_WORDS)

reduction_rate = 0.1  #defines how small the output summary should be compared with the input

text = """I saw ABC back in Neuro-Oncology Clinic today. He comes in for an urgent visit because of increasing questions about what to do next for his anaplastic astrocytoma.
Within the last several days, he has seen you in clinic and once again discussed whether or not to undergo radiation for his left temporal lesion. The patient has clearly been extremely ambivalent about this therapy for reasons that are not immediately apparent. It is clear that his MRI is progressing and that it seems unlikely at this time that anything other than radiation would be particularly effective. Despite repeatedly emphasizing this; however, the patient still is worried about potential long-term side effects from treatment that frankly seem unwarranted at this particular time.
After seeing you in clinic, he and his friend again wanted to discuss possible changes in the chemotherapy regimen. They came in with a list of eight possible agents that they would like to be administered within the next two weeks. They then wanted another MRI to be performed and they were hoping that with the use of this type of approach, they might be able to induce another remission from which he can once again be spared radiation.
From my view, I noticed a man whose language has deteriorated in the week since I last saw him. This is very worrisome. Today, for the first time, I felt that there was a definite right facial droop as well. Therefore, there is no doubt that he is becoming symptomatic from his growing tumor. It suggests that he is approaching the end of his compliance curve and that the things may rapidly deteriorate in the near future.
Emphasizing this once again, in addition, to recommending steroids I once again tried to convince him to undergo radiation. Despite an hour, this again amazingly was not possible. It is not that he does not want treatment, however. Because I told him that I did not feel it was ethical to just put him on the radical regimen that him and his friend devised, we compromised and elected to go back to Temodar in a low dose daily type regimen. We would plan on giving 75 mg/sq m everyday for 21 days out of 28 days. In addition, we will stop thalidomide 100 mg/day. If he tolerates this for one week, we then agree that we would institute another one of the medications that he listed for us. At this stage, we are thinking of using Accutane at that point.
While I am very uncomfortable with this type of approach, I think as long as he is going to be monitored closely that we may be able to get away with this for at least a reasonable interval. In the spirit of compromise, he again consented to be evaluated by radiation and this time, seemed more resigned to the fact that it was going to happen sooner than later. I will look at this as a positive sign because I think radiation is the one therapy from which he can get a reasonable response in the long term.
I will keep you apprised of followups. If you have any questions or if I could be of any further assistance, feel free to contact me."""


# !pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz

nlp_pl = spacy.load('en_core_web_sm')     #process original text according with the Spacy nlp pipeline for english
document = nlp_pl(text)                   #doc object

tokens = [token.text for token in document] #tokenized text

word_frequencies = {}
for word in document:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

max_frequency = max(word_frequencies.values())
print(max_frequency)

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency

sentence_tokens = [sent for sent in document.sents]

def get_sentence_scores(sentence_tok, len_norm=True):
    sentence_scores = {}
    i = 0
    for sent in sentence_tok:
        i += 1
        word_count = 0
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                word_count += 1
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
        if len_norm:
            try :
                sentence_scores[sent] = sentence_scores[sent]/word_count
            except:
                pass
    return sentence_scores
sentence_scores = get_sentence_scores(sentence_tokens,len_norm=False)        
sentence_scores_rel = get_sentence_scores(sentence_tokens) 

def get_summary(sentence_sc, rate):
    summary_length = int(len(sentence_sc)*rate)
    summary = nlargest(summary_length, sentence_sc, key = sentence_sc.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary

print("- NON_REL: "+ get_summary(sentence_scores, reduction_rate))
print("- REL: "+ get_summary(sentence_scores_rel, reduction_rate))

s1 = []


for text in transcription:
    nlp_pl = spacy.load('en_core_web_sm')     #process original text according with the Spacy nlp pipeline for english
    document = nlp_pl(text)                   #doc object

    tokens = [token.text for token in document] #tokenized text

    word_frequencies = {}
    for word in document:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    # print(word_frequencies)
    sentence_tokens = [sent for sent in document.sents]
    sentence_scores = get_sentence_scores(sentence_tokens,len_norm=False)        
    sentence_scores_rel = get_sentence_scores(sentence_tokens) 
    
    s1.append(get_summary(sentence_scores, reduction_rate))


s2 = []

from summarizer import Summarizer

model = Summarizer()

summary_length = int(len(sentence_tokens)*reduction_rate)

# for text in transcription:
#     result = model(text, num_sentences=summary_length, min_length=60)
#     s2.append(''.join(result))

model(text, num_sentences=summary_length, min_length=60)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

s3 = []
summarizer(text,max_length=1024, min_length=30, do_sample=False)



tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

# # by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
# model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")
# model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", attention_type="original_full")
# model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", block_size=16, num_random_blocks=2)

model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", attention_type="original_full")

inputs = tokenizer(text, return_tensors='pt')
prediction = model.generate(**inputs)
prediction = tokenizer.batch_decode(prediction)
# get_ipython().system('pip3 install summa')


s4 = []

summarizer.summarize(text)

for text in transcription:
    s4.append(summarizer.summarize(text))


ROUGE = Rouge()

reference = 'He comes in for an urgent visit because of increasing questions about what to do next for his anaplastic astrocytoma. Emphasizing this once again, in addition, to recommending steroids I once again tried to convince him to undergo radiation. If he tolerates this for one week, we then agree that we would institute another one of the medications that he listed for us.'

r1, r2, r3, r4 = [], [], [], []

r1[0]


for i in range(len(transcription)):
    r1.append(ROUGE.get_scores(s1[i], reference[i]))
    r2.append(ROUGE.get_scores(s2[i], reference[i]))
    r4.append(ROUGE.get_scores(s4[i], reference[i]))


ro1,ro2,rol = [0,0,0],[0,0,0],[0,0,0]
for i in r1:
    ro1[0] += i[0]['rouge-1']['r']
    ro1[1] += i[0]['rouge-1']['p']
    ro1[2] += i[0]['rouge-1']['f']
    
    ro2[0] += i[0]['rouge-2']['r']
    ro2[1] += i[0]['rouge-2']['p']
    ro2[2] += i[0]['rouge-2']['f']
    
    rol[0] += i[0]['rouge-l']['r']
    rol[1] += i[0]['rouge-l']['p']
    rol[2] += i[0]['rouge-l']['f']
for i in range(3):
    ro1[i] = round(ro1[i]/24,3)
    ro2[i] = round(ro2[i]/24,3)
    rol[i] = round(rol[i]/24,3)
print('R1 : ', ro1)
print('R1 : ', ro2)
print('R1 : ', rol)


ro1,ro2,rol = [0,0,0],[0,0,0],[0,0,0]
for i in r2:
    ro1[0] += i[0]['rouge-1']['r']
    ro1[1] += i[0]['rouge-1']['p']
    ro1[2] += i[0]['rouge-1']['f']
    
    ro2[0] += i[0]['rouge-2']['r']
    ro2[1] += i[0]['rouge-2']['p']
    ro2[2] += i[0]['rouge-2']['f']
    
    rol[0] += i[0]['rouge-l']['r']
    rol[1] += i[0]['rouge-l']['p']
    rol[2] += i[0]['rouge-l']['f']
for i in range(3):
    ro1[i] = round(ro1[i]/24,3)
    ro2[i] = round(ro2[i]/24,3)
    rol[i] = round(rol[i]/24,3)
print('R2 : ', ro1)
print('R2 : ', ro2)
print('R2 : ', rol)


ro1,ro2,rol = [0,0,0],[0,0,0],[0,0,0]
for i in r4:
    ro1[0] += i[0]['rouge-1']['r']
    ro1[1] += i[0]['rouge-1']['p']
    ro1[2] += i[0]['rouge-1']['f']
    
    ro2[0] += i[0]['rouge-2']['r']
    ro2[1] += i[0]['rouge-2']['p']
    ro2[2] += i[0]['rouge-2']['f']
    
    rol[0] += i[0]['rouge-l']['r']
    rol[1] += i[0]['rouge-l']['p']
    rol[2] += i[0]['rouge-l']['f']
for i in range(3):
    ro1[i] = round(ro1[i]/24,3)
    ro2[i] = round(ro2[i]/24,3)
    rol[i] = round(rol[i]/24,3)
print('R4 : ', ro1)
print('R4 : ', ro2)
print('R4 : ', rol)

ro1
ro1



