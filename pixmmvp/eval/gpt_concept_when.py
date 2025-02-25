from tqdm import tqdm
import argparse
import json
import openai
import re
import time
import numpy as np
# Create the parser
parser = argparse.ArgumentParser(description='Process OpenAI API key and JSONL file path.')

# Add arguments
parser.add_argument('--openai_api_key', default = "", help='Your OpenAI API key')
parser.add_argument('--concept_file', default = "concepts.npy",help='Path to the npy file')

# Parse arguments
args = parser.parse_args()

openai.api_key = args.openai_api_key
NUM_SECONDS_TO_SLEEP = 10
# Define a function to query the OpenAI API and evaluate the answer
def retrieve_concept(question):
    while True:
        try:
            response = openai.ChatCompletion.create(
                #model='gpt-4-0314',
                model='gpt-4o',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for providing the main concept of the noun phrase of interest.'
                }, {
                    'role': 'user',
                    'content': question,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response['choices'][0]['message']['content']
    print(answer)
    return answer.lower()

#concepts_answer = []
#noun_phrases_metainfo = np.load(args.concept_file, allow_pickle=True)
#question = 'Can you group these noun phrases into categories of abstract concepts. You are provided with pairs of (noun phrase, its full context);'
#
#idx = 0
#for noun_phr, full_text in noun_phrases_metainfo:
#    idx += 1
#    if idx == 10:
#        idx = 0
#        concepts_answer.append(retrieve_concept(question))
#        question = 'Can you group these noun phrases into categories of abstract concepts. You are provided with pairs of (noun phrase, its full context);'
#
#    question += "('%s', '%s') "%(noun_phr, full_text)
#np.save(args.concept_file.split('.')[0]+'_gptconcepts.npy', concepts_answer)

#concepts_gpt = np.load(args.concept_file, allow_pickle=True)
#concepts_final = [st for phr in concepts_gpt for st in phr.split("**") if ':' in st and 'Certainly' not in st and '\n' not in st]
#question = 'Can you group these noun categories into only 9 categories of abstract concepts: '
#for concept in concepts_final:
#    question  += concept.replace(':', ',')+' '
#print(retrieve_concept(question))

#concepts_answer = []
#noun_phrases_metainfo = np.load(args.concept_file, allow_pickle=True)
#question = 'Can you classify the following noun phrase belongs to which of these 6 concepts: (a) Color and appearance, (b) Location and position, (c) Object part, (d) Context or setting, (e) Objects/entities, (f) State. Please respond with one or two letters and provide letters only. You will be given the noun phrase and the correponding full context.'
#
#final_concept_cats = []
#idx = 0
#for noun_phr, full_text in noun_phrases_metainfo:
#    concept_cat = retrieve_concept(question+f': \'{noun_phr}\', \'{full_text}\'. ')
#    final_concept_cats.append(concept_cat.lower())
#
#np.save(args.concept_file.split('.npy')[0]+'_finalconceptcats.npy', final_concept_cats)

concept_cats = np.load(args.concept_file, allow_pickle=True)
nbins = 6
bins = {chr(i+ord('a')): 0 for i in range(nbins)}

for concept_cat in concept_cats:
    cats = concept_cat.split(',')
    for cat in cats:
        cat = cat.replace('(','').replace(')','')
        try:
            bins[cat.strip()] += 1
        except:
            cat = cat.split(' ')[0]
            bins[cat.strip()] += 1

print(bins)
