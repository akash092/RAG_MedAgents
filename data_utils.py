import os
import jsonlines
import re
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

class MyDataset:
    def __init__(self, split, args, eval_only=False, traindata_obj=None):
        #self.counter = 0
        if hasattr(args, 'start_pos'):
            self.start_pos = args.start_pos
        if hasattr(args, 'end_pos'):
            self.end_pos = args.end_pos
        if hasattr(args, 'model_name'):
            self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.dir_path = args.dataset_dir
        self.split = split  # train / test
        self.load() # load dataset -> load data in self.data

    def load(self): # load dataset -> self.data
        filename = os.path.join(self.dir_path, self.split + '.jsonl')
        self.data = []
        with open(filename) as f:
            for item in jsonlines.Reader(f):
                self.data.append(item)

    def get_by_idx(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def remove_incomplete_sentence(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-1][-1] != '.':
        return ' '.join(sentences[:-1]) + '.'   #remove the last sentence
    else:
        return text

def cleansing_analysis(analyses, domains, type):
    analysis = {}
    
    for i, item in enumerate(analyses):
        if item == "ERROR.":
            item = f"There is no analysis for this {type}."
        item = remove_incomplete_sentence(item)
        if "as an ai language model" in item.lower():
            end_index = item.lower().find("as an ai language model")+len("as an ai language model")
            item= item[end_index:].strip().strip(',').strip()
        analysis[domains[i]] = item
    
    return analysis


def cleansing_syn_report(question, options, raw_synthesized_report):

    tmp = raw_synthesized_report.split("Total Analysis:")
    total_analysis_text = tmp[1].strip()
    if "Key Knowledge" in tmp:
        key_knowledge_text = tmp[0].split("Key Knowledge:")[-1].strip()
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Key Knowledge: {key_knowledge_text} \n" \
            f"Total Analysis: {total_analysis_text} \n"
    else:
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Total Analysis: {total_analysis_text} \n"
    
    return final_syn_repo

def cleansing_final_output(output):
    try:
        ans = output.split(":")[-1]
        ans = re.findall(r'A|B|C|D|E', ans)
        if len(ans) == 0:
            ans = ""
        else:
            ans = ans[0]
    except:
        ans = re.findall(r'A|B|C|D|E', ans)
        if len(ans) == 0:
            ans = ""
        else:
            ans = ans[-1]
    
    return ans, output

def cleansing_voting(output):
    output = output.lower()
    ans = re.findall(r'yes|no', output)
    if len(ans) == 0:
        ans = "yes"
    else:
        ans = ans[0]
    return ans


def transform_dict2text(analyses, type, content):
    if type == "question":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
            report += f"Report{i} \n" \
                f"Question: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1
    elif type == "options":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
            report += f"Report{i}: \n" \
                f"Options: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1
    return report
