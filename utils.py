from prompt_generator import *
from data_utils import *
from retrieval_agent.rag_agent import RAG_AGENT_DOMAIN
def fully_decode(question, options, gold_answer, handler, dataset_name, max_attempt_vote, add_rag, rag_agent):

    question_domains, options_domains, question_analyses, option_analyses, syn_report, output = "", "", "", "", "", ""
    vote_history, revision_history, syn_repo_history = [], [], []

    # get question domains
    question_classifier, prompt_get_question_domain = get_question_domains_prompt(question)
    raw_question_domain = handler.get_output_multiagent(user_input=prompt_get_question_domain, max_tokens=50, system_role=question_classifier)
    if raw_question_domain == "ERROR.":
        raw_question_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_QD)])
    question_domains = raw_question_domain.split(":")[-1].strip().split(" | ")


    if dataset_name == 'MedQA':
        # get option domains
        options_classifier, prompt_get_options_domain = get_options_domains_prompt(question, options)
        raw_option_domain = handler.get_output_multiagent(user_input=prompt_get_options_domain, max_tokens=50, system_role=options_classifier)
        if raw_option_domain == "ERROR.":
            raw_option_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_OD)])
        options_domains = raw_option_domain.split(":")[-1].strip().split(" | ")
    else:
        options_domains = []

    # get question analysis
    tmp_question_analysis = []
    for _domain in question_domains:
        question_analyzer, prompt_get_question_analysis = get_question_analysis_prompt(question, _domain)
        raw_question_analysis = handler.get_output_multiagent(user_input=prompt_get_question_analysis, max_tokens=300, system_role=question_analyzer)
        tmp_question_analysis.append(raw_question_analysis)
    if add_rag:
        question_domains.append(RAG_AGENT_DOMAIN)
        raw_question_analysis = rag_agent.chat(question)
        tmp_question_analysis.append(raw_question_analysis)
    question_analyses = cleansing_analysis(tmp_question_analysis, question_domains, 'question')


    if dataset_name == 'MedQA':
        # get option analysis if applicable
        tmp_option_analysis = []
        for _domain in options_domains:
            option_analyzer, prompt_get_options_analyses = get_options_analysis_prompt(question, options, _domain, question_analyses)
            raw_option_analysis = handler.get_output_multiagent(user_input=prompt_get_options_analyses, max_tokens=300, system_role=option_analyzer)
            tmp_option_analysis.append(raw_option_analysis)
        option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, 'option')
    else:
        option_analyses = {}

    # get synthesized report
    q_analyses_text = transform_dict2text(question_analyses, "question", question)
    o_analyses_text = transform_dict2text(option_analyses, "options", options)
    synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(q_analyses_text, o_analyses_text)
    raw_synthesized_report = handler.get_output_multiagent(user_input=prompt_get_synthesized_report, max_tokens=2500, system_role=synthesizer)
    syn_report = cleansing_syn_report(question, options, raw_synthesized_report)

    # collaborative consultation
    all_domains = question_domains + options_domains
    syn_repo_history = [syn_report]
    hasno_flag = True   # default value: in order to get into the while loop
    num_try = 0

    while num_try < max_attempt_vote and hasno_flag:
        domain_opinions = {}    # 'domain' : 'yes' / 'no'
        revision_advice = {}
        num_try += 1
        hasno_flag = False
        # hold a meeting for all domain experts to vote and gather advice if they do not agree
        for domain in all_domains:
            voter, cons_prompt = get_consensus_prompt(domain, syn_report)
            raw_domain_opi = handler.get_output_multiagent(user_input=cons_prompt, max_tokens=30, system_role=voter)
            domain_opinion = cleansing_voting(raw_domain_opi)   # "yes" / "no"
            domain_opinions[domain] = domain_opinion
            if domain_opinion == "no":
                advice_prompt = get_consensus_opinion_prompt(domain, syn_report)
                advice_output = handler.get_output_multiagent(user_input=advice_prompt, max_tokens=500, system_role=voter)
                revision_advice[domain] = advice_output
                hasno_flag = True
        if hasno_flag:
            revision_prompt = get_revision_prompt(syn_report, revision_advice)
            revised_analysis = handler.get_output_multiagent(user_input=revision_prompt, max_tokens=2500, system_role="")
            syn_report = cleansing_syn_report(question, options, revised_analysis)
            revision_history.append(revision_advice)
            syn_repo_history.append(syn_report)
        vote_history.append(domain_opinions)
    
    # final answer derivation
    answer_prompt = get_final_answer_prompt_wsyn(syn_report)
    output = handler.get_output_multiagent(user_input=answer_prompt, max_tokens=2500, system_role="")
    # TODO: cleansing on options based on available option in the problem
    ans, output = cleansing_final_output(output)
                        


    data_info = {
        'question': question,
        'options': options,
        'pred_answer': ans,
        'gold_answer': gold_answer,
        'question_domains': question_domains,
        'option_domains': options_domains,
        'question_analyses': question_analyses,
        'option_analyses': option_analyses,
        'syn_report': syn_report,
        'vote_history': vote_history,
        'revision_history': revision_history,
        'syn_repo_history': syn_repo_history,
        'raw_output': output
    }
    
    return data_info

