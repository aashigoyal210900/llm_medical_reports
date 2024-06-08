!pip install transformers
!pip install fastai.text.all
!pip install ohmeow-blurr -q
!pip install bert-score -q
!pip install torch==2.0.0


import torch
from transformers import *
from fastai.text.all import *

from blurr.text.data.all import *
from blurr.text.modeling.all import *

df = pd.read_csv('mtsamples.csv')
df = df.drop(columns=['Unnamed: 0', 'medical_specialty','sample_name','keywords'])
df.isnull().sum()
df = df.dropna().reset_index()
df = df.drop(columns=['index'])

#Clean text
df['transcription'] = df['transcription'].apply(lambda x: x.replace('\n',''))

#Select only part of it (makes testing faster)
articles = df.head(500)
articles.head()

pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(pretrained_model_name, model_cls=BartForConditionalGeneration)

hf_arch, type(hf_config), type(hf_tokenizer), type(hf_model)

text_gen_kwargs = {}
if hf_arch in ["bart", "t5"]:
    text_gen_kwargs = {**hf_config.task_specific_params["summarization"], **{"max_length": 30, "min_length": 10}}

# not all "summarization" parameters are for the model.generate method ... remove them here
generate_func_args = list(inspect.signature(hf_model.generate).parameters.keys())
for k in text_gen_kwargs.copy():
    if k not in generate_func_args:
        del text_gen_kwargs[k]

if hf_arch == "mbart":
    text_gen_kwargs["decoder_start_token_id"] = hf_tokenizer.get_vocab()["en_XX"]

tok_kwargs = {}
if hf_arch == "mbart":
    tok_kwargs["src_lang"], tok_kwargs["tgt_lang"] = "en_XX", "en_XX"

# hf_batch_tfm = Seq2SeqBatchTokenizeTransform(hf_arch, hf_config, hf_tokenizer, hf_model, 
#     task='summarization',
#     text_gen_kwargs=
#  {'max_length': 248,'min_length': 56,'do_sample': False, 'early_stopping': True, 'num_beams': 4, 'temperature': 1.0, 
#   'top_k': 50, 'top_p': 1.0, 'repetition_penalty': 1.0, 'bad_words_ids': None, 'bos_token_id': 0, 'pad_token_id': 1,
#  'eos_token_id': 2, 'length_penalty': 2.0, 'no_repeat_ngram_size': 3, 'encoder_no_repeat_ngram_size': 0,
#  'num_return_sequences': 1, 'decoder_start_token_id': 2, 'use_cache': True, 'num_beam_groups': 1,
#  'diversity_penalty': 0.0, 'output_attentions': False, 'output_hidden_states': False, 'output_scores': False,
#  'return_dict_in_generate': False, 'forced_bos_token_id': 0, 'forced_eos_token_id': 2, 'remove_invalid_values': False})

hf_batch_tfm = Seq2SeqBatchTokenizeTransform(
    hf_arch,
    hf_config,
    hf_tokenizer,
    hf_model,
    max_length=256,
    max_target_length=130,
    tok_kwargs=tok_kwargs,
    text_gen_kwargs=text_gen_kwargs,
)

#Prepare data for training
blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=hf_batch_tfm), noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader('transcription'), get_y=ColReader('description'), splitter=RandomSplitter())

dls = dblock.dataloaders(articles, batch_size = 2,num_workers=0)

seq2seq_metrics = {
    "rouge": {
        "compute_kwargs": {"rouge_types": ["rouge1", "rouge2", "rougeL", "rougeLsum"], "use_stemmer": True},
        "returns": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    },
    "bertscore": {
        "compute_kwargs": {"lang": "en"}, 
        "returns": ["precision", "recall", "f1"]},
}

model = BaseModelWrapper(hf_model)
learn_cbs = [BaseModelCallback]
fit_cbs = [Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

learn = Learner(
    dls,
    model,
    opt_func=partial(Adam),
    loss_func=CrossEntropyLossFlat(),
    cbs=learn_cbs,
    splitter=partial(blurr_seq2seq_splitter, arch=hf_arch),
)

learn.create_opt() 
learn.freeze()

import nltk
nltk.download('punkt')

learn.fit_one_cycle(1, lr_max=4e-5, cbs=fit_cbs)

text_to_generate =  """Thank you for referring Mr. Sample Patient for cardiac evaluation. This is a 67-year-old, obese male who has a history of therapy-controlled hypertension, borderline diabetes, and obesity. He has a family history of coronary heart disease but denies any symptoms of angina pectoris or effort intolerance. Specifically, no chest discomfort of any kind, no dyspnea on exertion unless extreme exertion is performed, no orthopnea or PND. He is known to have a mother with coronary heart disease. He has never been a smoker. He has never had a syncopal episode, MI, or CVA. He had his gallbladder removed. No bleeding tendencies. No history of DVT or pulmonary embolism. The patient is retired, rarely consumes alcohol and consumes coffee moderately. He apparently has a sleep disorder, according to his wife (not in the office), the patient snores and stops breathing during sleep. He is allergic to codeine and aspirin (angioedema).
Physical exam revealed a middle-aged man weighing 283 pounds for a height of 5 feet 11 inches. His heart rate was 98 beats per minute and regular. His blood pressure was 140/80 mmHg in the right arm in a sitting position and 150/80 mmHg in a standing position. He is in no distress. Venous pressure is normal. Carotid pulsations are normal without bruits. The lungs are clear. Cardiac exam was normal. The abdomen was obese and organomegaly was not palpated. There were no pulsatile masses or bruits. The femoral pulses were 3+ in character with a symmetrical distribution and dorsalis pedis and posterior tibiales were 3+ in character. There was no peripheral edema.
He had a chemistry profile, which suggests diabetes mellitus with a fasting blood sugar of 136 mg/dl. Renal function was normal. His lipid profile showed a slight increase in triglycerides with normal total cholesterol and HDL and an acceptable range of LDL. His sodium was a little bit increased. His A1c hemoglobin was increased. He had a spirometry, which was reported as normal.
He had a resting electrocardiogram on December 20, 2002, which was also normal. He had a treadmill Cardiolite, which was performed only to stage 2 and was terminated by the supervising physician when the patient achieved 90% of the predicted maximum heart rate. There were no symptoms or ischemia by EKG. There was some suggestion of inferior wall ischemia with normal wall motion by Cardiolite imaging.
In summary, we have a 67-year-old gentleman with risk factors for coronary heart disease. I am concerned with possible diabetes and a likely metabolic syndrome of this gentleman with truncal obesity, hypertension, possible insulin resistance, and some degree of fasting hyperglycemia, as well as slight triglyceride elevation. He denies any symptoms of coronary heart disease, but he probably has some degree of coronary atherosclerosis, possibly affecting the inferior wall by functional testings.
In view of the absence of symptoms, medical therapy is indicated at the present time, with very aggressive risk factor modification. I explained and discussed extensively with the patient, the benefits of regular exercise and a walking program was given to the patient. He also should start aggressively losing weight. I have requested additional testing today, which will include an apolipoprotein B, LPa lipoprotein, as well as homocystine, and cardio CRP to further assess his risk of atherosclerosis.
In terms of medication, I have changed his verapamil for a long acting beta-blocker, he should continue on an ACE inhibitor and his Plavix. The patient is allergic to aspirin. I also will probably start him on a statin, if any of the studies that I have recommended come back abnormal and furthermore, if he is confirmed to have diabetes. Along this line, perhaps, we should consider obtaining the advice of an endocrinologist to decide whether this gentleman needs treatment for diabetes, which I believe he should. This, however, I will leave entirely up to you to decide. If indeed, he is considered to be a diabetic, a much more aggressive program should be entertained for reducing the risks of atherosclerosis in general, and coronary artery disease in particular.
I do not find an indication at this point in time to proceed with any further testing, such as coronary angiography, in the absence of symptoms."""

outputs = learn.blurr_generate(text_to_generate, early_stopping=False, num_return_sequences=3)

for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')


text = """I saw ABC back in Neuro-Oncology Clinic today. He comes in for an urgent visit because of increasing questions about what to do next for his anaplastic astrocytoma.
Within the last several days, he has seen you in clinic and once again discussed whether or not to undergo radiation for his left temporal lesion. The patient has clearly been extremely ambivalent about this therapy for reasons that are not immediately apparent. It is clear that his MRI is progressing and that it seems unlikely at this time that anything other than radiation would be particularly effective. Despite repeatedly emphasizing this; however, the patient still is worried about potential long-term side effects from treatment that frankly seem unwarranted at this particular time.
After seeing you in clinic, he and his friend again wanted to discuss possible changes in the chemotherapy regimen. They came in with a list of eight possible agents that they would like to be administered within the next two weeks. They then wanted another MRI to be performed and they were hoping that with the use of this type of approach, they might be able to induce another remission from which he can once again be spared radiation.
From my view, I noticed a man whose language has deteriorated in the week since I last saw him. This is very worrisome. Today, for the first time, I felt that there was a definite right facial droop as well. Therefore, there is no doubt that he is becoming symptomatic from his growing tumor. It suggests that he is approaching the end of his compliance curve and that the things may rapidly deteriorate in the near future.
Emphasizing this once again, in addition, to recommending steroids I once again tried to convince him to undergo radiation. Despite an hour, this again amazingly was not possible. It is not that he does not want treatment, however. Because I told him that I did not feel it was ethical to just put him on the radical regimen that him and his friend devised, we compromised and elected to go back to Temodar in a low dose daily type regimen. We would plan on giving 75 mg/sq m everyday for 21 days out of 28 days. In addition, we will stop thalidomide 100 mg/day. If he tolerates this for one week, we then agree that we would institute another one of the medications that he listed for us. At this stage, we are thinking of using Accutane at that point.
While I am very uncomfortable with this type of approach, I think as long as he is going to be monitored closely that we may be able to get away with this for at least a reasonable interval. In the spirit of compromise, he again consented to be evaluated by radiation and this time, seemed more resigned to the fact that it was going to happen sooner than later. I will look at this as a positive sign because I think radiation is the one therapy from which he can get a reasonable response in the long term.
I will keep you apprised of followups. If you have any questions or if I could be of any further assistance, feel free to contact me."""

export_fname = "summarize_export_final"

learn.metrics = None
learn = learn.to_fp32()
learn.export(fname=f"{export_fname}.pkl")

inf_learn = load_learner(fname=f"{export_fname}.pkl")
inf_learn.blurr_summarize(text)

