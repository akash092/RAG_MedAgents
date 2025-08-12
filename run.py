from data_utils import MyDataset
from api_utils import api_handler
from string import punctuation
import argparse
import tqdm
import json
from utils import *
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from retrieval_agent.rag_agent import RAGAgent
import random 

def shuffle_dict_values(original_dict):
    """
    Shuffles the values of a dictionary and assigns them to different keys
    """
    if not original_dict:
        return {}

    keys = list(original_dict.keys())
    values = list(original_dict.values())

    # Shuffle the values
    random.shuffle(values)

    # Create the new dictionary with original keys and shuffled values
    shuffled_dict = dict(zip(keys, values))
    return shuffled_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt4.1')
    parser.add_argument('--dataset_name', default='MedQA')
    parser.add_argument('--dataset_dir', default='./datasets/MedQA/')
    parser.add_argument('--start_pos', type=int, default=6)
    parser.add_argument('--end_pos', type=int, default=7)
    parser.add_argument('--output_files_folder', default='.')
    parser.add_argument('--use_rag_agent', type=bool, default=False)

    parser.add_argument('--max_attempt_vote', type=int, default=3)
    parser.add_argument('--do_choice_shuffling', type=bool, default=False)
    args = parser.parse_args()

    print(args)

    ### get handler
    if args.model_name in ['instructgpt', 'newinstructgpt', 'chatgpt', 'gpt4.1']: # select the model
        handler = api_handler(args.model_name)
    else:
        raise ValueError

    retriever = FAISS.load_local("retrieval_agent/faiss_index/", OpenAIEmbeddings(), allow_dangerous_deserialization=True).as_retriever()
    # Create agents
    rag_agent = RAGAgent(retriever=retriever, handler = handler)

    ### get dataobj
    dataobj = MyDataset('test', args, traindata_obj=None)

    ### set test range
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)  # closed interval

    ### set output_file_name
    exact_output_file = f"{args.output_files_folder}/{args.dataset_name}_{args.model_name}.json"

    total_questions = 0
    correct = 0
    input_prompt = {}
    with open(exact_output_file, 'a') as f:
        f.write("[\n")  # Start the JSON array
        for idx in tqdm.tqdm(test_range, desc=f"{args.start_pos} ~ {end_pos}"):
            raw_sample = dataobj.get_by_idx(idx)
            question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'            
            
            options = raw_sample['options']
            gold_answer = raw_sample['answer_idx']
            gold_answer_string = options[gold_answer]
            
            if args.do_choice_shuffling:
                gold_answer = ""
                options = shuffle_dict_values(options)
                for k, v in options.items():
                    if v == gold_answer_string:
                        gold_answer = k
                        break
            
            if gold_answer == "":
                raise ValueError("gold_answer is empty")
            
            data_info = fully_decode(question, options, gold_answer, handler, args.dataset_name, args.max_attempt_vote, args.use_rag_agent, rag_agent)
            
            record = json.dumps(data_info)
            if idx == end_pos -1:
                f.write(record + '\n')
            else:
                f.write(",\n")
        
        
        f.write("]\n")  # Close the JSON array
        
    # compute accuracy
    total_questions += 1
    if data_info["pred_answer"] == gold_answer:
        correct += 1

    accuracy_percentage = (correct*100)/total_questions
    print(f"{args.use_rag_agent=}, {args.do_choice_shuffling}")
    print(f"{total_questions=}, {accuracy_percentage=}, {args.dataset_name=}, ")