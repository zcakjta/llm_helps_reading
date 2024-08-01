
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
import requests
from typing import TypedDict
from langgraph.graph import StateGraph
import os
import ast
import json
from dotenv import load_dotenv
os.environ['LANGCHAIN_PROJECT'] = 'Langgraph'
load_dotenv()


# make calls to pinecone database
def get_full_docs(user_query, top_k=3):
    embd = OpenAIEmbeddings(model="text-embedding-3-small")
    Pinecone_api_key = os.getenv("PINECONE_API_KEY)")
    pc = Pinecone(api_key=Pinecone_api_key)
    index = pc.Index("ai-pm")
    query_vector = embd.embed_query(user_query)
    query_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    def fetch_all_chunks(base_id, total_chunks):
        ids = [f"{base_id}_{i}" for i in range(total_chunks)]
        all_chunks = index.fetch(ids)
        return all_chunks

    def reconstruct_metadata(all_chunks):
        reconstructed_metadata = {}
        full_text = ""
        for chunk in all_chunks.vectors.values():
            full_text += chunk.metadata.get('text', '') + '\n\n'
        reconstructed_metadata = {'text': full_text.strip()}  # Remove the last newline
        return reconstructed_metadata

    full_docs = []
    full_id = []

    for result in query_results.matches:
        id = result['id']
        #print(id)
        full_id.append(id)
        if id.startswith('doc_'):
            base_id = result['metadata']['base_id']
            #print(base_id)
            total_chunks = int(result['metadata']['total_chunks'])
            #print(total_chunks)

            all_chunks = fetch_all_chunks(base_id, total_chunks)
            reconstructed_metadata = reconstruct_metadata(all_chunks)

            # Access the text key from the reconstructed metadata
            full_combined_text = reconstructed_metadata.get('text', '')
            full_docs.append(full_combined_text)

        elif id.startswith('full_'):
            full_text = result['metadata']['text']
            full_docs.append(full_text)
            print(f"Full text for {id}: {full_text}")
    full_docs_str = "\n\n".join(doc for doc in full_docs)
    return full_docs


llm = ChatOpenAI(model = "gpt-4o-mini")    

# analyze the questions
class AIpm_analyze:
    def __int__(self):
        pass

    def analyze(self,summary:str):
        model = llm
        system = """ You are a research assistant who helps review research summaries, articles, and blogs.
        You will be given a acontext to review.
        Summaries may come from different sources, including interviews, talks, articles, blogs, research reports/
        Your need to extract the key information to help form a comprehensive understanding of this piece of condensed information.
        the below are 7 Ws to help your thought process\n

        - what : what is the information about ? what is discussed and what are the key findings and opinions\n
        - who : what is the information for. who is the target audience and user and market\n
        - why : what is the purpose of this information\n
        - how: how are the key findings and opinions formed ? how are these illustrated\n
        - by whom : by whom is this information meant to send \n
        - when : what is the time that this information is published. what else is happening in during the same period.\n
        - where : what is the context for this information, in what societies or markets \n
        
        Your should first produce a condensed summary using the above 7 Ws. This is to understand the core essence of the context provided
        Secondly , you need to produce a generalization of this context to derive high-level concepts and first principles. 
        This generalization will be used to retrieve broader information about the big picture around the key essences
        and for facilitating conducting literature review to understand what is already known about the big picture and what is novel about
        the details mentioned in the essence

        
        This is the general guidances that you should follow: \n
        1.Only use information provided in the summary for your thought process.
        2.identify key characters, entities, organizations mentioned. This is needed to identify the essence without confusion.
	    3.Return the response in json format like this 
        {{"condensed summary": "This is the output for the first part",
          " generalization" : "this is the output for the second part"}}
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                ("human",
                 "\n \
                  This is the summary:\n \
                    -----\n{summary}\n-----\n")
            ]
        )
        analyze_chain = prompt|llm|StrOutputParser()

        analyze_summary = analyze_chain.invoke({"summary":summary})

        return analyze_summary
    
    def run (self,state:dict):
        summary = state["summary"]
        analyze_summary = self.analyze(summary)

        state["analyze_summary"] = analyze_summary

        return state
    
# prepare for litearture review
class Aipm_prepare:
    def __init__(self):
        pass

    def prepare(self, analyze_summary):
        model = llm
        category =[
            "AI & Product Management and Strategy",
            "AI & State-of-the-art models and techs",
            "AI & first principle understanding",
            "AI & start-ups and VC investments",
            "AI & Industry insights and know-how"
        ]

        system = """You are a research assistant with expertise in AI-related topics.\n 
        You are given some extra readings as inspirations. 
        This is provided by your mentor who has already broken the information into two parts:\n
        The first part is a condensed summary of the reading.
        The second part is about the generalization abstracted that talks about the big picture and context background. It is useful for conducting literature review.

        Your task is to analyze the inforamtion given by your mentor in order to categorize the topic of the extra reading into the predefined fields
        Then based on the fields you have assigned, you need to add some of your thoughts and interests to ask more questions for literature review specfically focusing
        on what is already known in this particular filed
        
        RESPOND IN JSON FORMAT like below:
        {{"condensed summary" : " THIS IS THE CONDENSED SUMMARY ALREADY GIVEN BY YOUR MENTOR. You may add some of your own thoughts",
        "generalization" :" THIS IS THE GENRALIZATION ALREADY GIVEN BY YOUR MENTOR.",
        "related field" : " THIS IS INTERESTED FIELD YOU HAVE CHOSEN AND THIS WILL GUIDE YOUR FURTHER RESEARCH AND THOUGHTS FROM THIS PARTICULAR PERSPECTIVE"
        " thoughts" : "THIS IS WHERE YOU NEED TO ADD YOUR OWN THOUHTS FOR CONDUCTING LITERATURE REVIEW AND ADD QUESTIONS THAT YOU WANT TO FOCUS TO COMPARE WITH PREVIOUS STUDIES
        }}
        NOTICE THAT, SOMETIMES PROVIDED EXTRA READING MIGHT NOT RELATE TO AI. IF THAT IS THE CASE, YOU SHOULD ONLY REPLY WITH 'NOT RELATED TO AI'
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                ("human",
                 "You are given the following opinions.\n \
                  This is the information:\n \
                    -----\n{analyze_summary}\n-----\n \
                    CHOSE FROM THE BELOW FILED. ONLY CHOOSE ONE\n \
                        This is the field:\n\
                    -----\n{category}\n-----\n\
                    generate your response in Chinese")
            ]
        )
        prepare_chain = prompt|llm|StrOutputParser()
        results_lists =[]
       
        prepare_results = prepare_chain.invoke({"analyze_summary":analyze_summary,"category":category}) 

        return prepare_results
    
    def run (self,state:dict):
        analyze_summary = state["analyze_summary"]
        prepare_results = self.prepare(analyze_summary)
        # it is a list of dictionary.
        state["prepare_results"] = prepare_results
        return state
    
# review and query the db
class Aipm_review:
    def __int__(self):
        pass

    def review(self, prepare_results):
        model = llm

        system = """You are a AI expert with various experiences in VC investments, product management, AI research, and etc...
            Your task is to generate search queries to faciliate litearture review and help establish perspectives to genearte novel points.
            You will be given a piece of information, whicch consists of the following parts:
            1. the condensed summary of an extra reading with details
            2. generalization based on the extra reading so that you know the big picture and background context
            3. the interested field that further research or information digging should be focused on
            4. the thoughts given by a research assistant to suggest directions for further research and litearture review.
    
            You should generate detailed search queries that covering the details of the information, the big picture/ background, and the interested field for further information
            gathering.
            
            Respond with a list of search quries in a format like this:
            ['query 1', 'query 2', 'query 3', '....']

            EXTRA GUIDENCE:
            - MAKE SURE TO HAVE SEARCH QUERIES THAT COVER DIFFERENT LEVELS OF GRANULARITY SO THAT LITERATURE REVIEWING AND FURTHER RESEARCH IS COMPREHENSIVE
            - ONLY USE THE INFORMATION PROVIDED. 
            - RESPOND IN CHINESE
            - PRODUCE MINIMUM 2 QURIES, MAXIMUM 3 QUERIES.
            """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                ("human",
                 "This is the information : {prepare_results}")
            ]
        )
        review_chain = prompt|llm|StrOutputParser()
        follow_up_questions = review_chain.invoke({"prepare_results":prepare_results})
        return follow_up_questions



    def run (self,state:dict):
        prepare_results = state["prepare_results"]
        questions = self.review(prepare_results)
        state["questions"] = questions
        return state

# rag qa    
class Aipm_qa:
    def __init__(self, api_key_env_var='PINECONE_API_KEY', index_name='ai-pm', model_name='gpt-4o-mini'):
        # Load the .env file
        load_dotenv()
        # Get the value of the API key
        pinecone_api_key = os.getenv(api_key_env_var)
        # Initialize Pinecone and vector store
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.model = ChatOpenAI(model=model_name)
        self.embd = OpenAIEmbeddings()
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embd)
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def run(self,state:dict):
        ### HERE WE NEED TO RETRIEVE INFORMATION.
        questions = state['questions']
        questions_lists = ast.literal_eval(questions)
        retrieved_docs =[]
        for q in questions_lists:
            docs = get_full_docs(q,top_k=3)
            retrieved_docs.append(docs)
        system = """
            You are a research assistant and AI expert. You need to answer the questions , using the provided context to conduct a literature review
            to help understand what is already known and what is the baseline knowledge here about this particular context and topic.
            Your answer will help your teammate to determine the novelty/
            THIS is the questions:
            {questions}
            THIS is the context:
            {context}
            
            respond with details and in CHINESE
            YOUR RESPONSE SHOULD COVER ALL THE QUETIONS and form a comprehensive perspective that has different level of granularity.
            """
        flattened_docs = []

        for doc in retrieved_docs:
            if isinstance(doc, list):
                flattened_docs.extend(doc)  # Flatten the list
            else:
                flattened_docs.append(doc)
    # Ensure all items are strings
        flattened_docs = [str(doc) for doc in flattened_docs]
        context = '\n\n'.join ( doc for doc in flattened_docs)
        baseline_prompt = PromptTemplate.from_template(system)
        model = self.model
        form_baseline_chain = baseline_prompt|model|StrOutputParser()

        baseline_results = form_baseline_chain.invoke({"questions":questions,"context":context})
        state["baseline_results"] =baseline_results
    
        # Define the QA prompt
        QA = """You are an expert AI product manager with extensive industry experience.\n
        You are given an extra reading:\n
        ------\n
        {original_context}
        ------\n
        This is questions and perspectives to be considered
        {prepare_results}\n
        ------\n
        Having deeply thought about these questions, you have conducted literature reviews and combien with your past experiences and know-hows\n
        This is what you have put together, this is the baseline and previous studies that you should compare to:\n
        ------\n
        {baseline_results}\n
        ------\n
        Your task:
        Based on the questions given and the experiences & know-hows you have. respond to the question: what are the 10 
        innovative points here mentioned by the recent AI news and why?
        Respond to the task with comprehensive details and illustrate with appropriate examples in Chinese.
        """

        # Initialize the prompt template
        self.prompt = PromptTemplate.from_template(QA)
        self.retriever_pinecone = self.vector_store.as_retriever()
        # Chain
        rag_chain = (
            self.prompt
            | self.model
            | StrOutputParser()
        )
        original_context = state["original_context"]
        prepare_results = state["prepare_results"]
        baseline_results = state["baseline_results"]
        response = rag_chain.invoke({"original_context":original_context,"prepare_results":prepare_results,"baseline_results":baseline_results})

        state['novel_points'] = response
    
        return state

# define the state
class Aipm_state(TypedDict):
    original_context: str
    summary:str
    analyze_summary:str
    baseline_results:str
    prepare_results:str
    questions:str
    original_context:str
    retrieved_docs:str
    novel_points:str
    
# master Agent
class AipmAgent:
    def __init__(self):
        pass
    
    def init_team(self):                                                                                            
        analyzer = AIpm_analyze()
        preparer=Aipm_prepare()
        reviewer = Aipm_review()
        qa = Aipm_qa()

        workflow = StateGraph(Aipm_state)

        workflow.add_node("analyzer",analyzer.run)
        workflow.add_node("preparer",preparer.run)
        workflow.add_node("reviewer",reviewer.run)
        workflow.add_node("qa",qa.run)

        workflow.add_edge("analyzer","preparer")
        workflow.add_edge("preparer","reviewer")
        workflow.add_edge("reviewer","qa")

        workflow.set_entry_point("analyzer")
        workflow.set_finish_point("qa")

        Aipm_graph = workflow.compile()

        return Aipm_graph
    
    def run(self):
        Aipm_graph = self.init_team()
        #def enter_chain(summary,orignal_context) :
            #return {"summary": summary, "original_context":orignal_context}
        
        Aipm_chain = Aipm_graph
        #Aipm_chain = enter_chain|Aipm_graph
        
        return Aipm_chain  


# further QA chain
class further_qa:
    def __init__(self):
        pass

    def question (self, summary):
        q_prompt = """ You are a great disruptive thinker in tech industry with extensive experiences in AI, Product Management, and venture capital investment.
        You are given a summary of an article. Just like how you read, evaluate, and absorb knowledge everyday.
        You review the opinions and events discussed in the summary, and you conduct critical thinking process and come up with questions in your mind
        Your Task is to generate 3 follow-up questions to help yourself think in depth and evaluate the information.\n
        ------\n
        This is the summary:\n{summary}\n
        ------\n
        ## General Guidance:
        - ensure 3 follow-up questions are highly relevant to the summary, based on key findings and the implications of the summary
        - questions should be critical-thinking, specific, and thoughtful 
        - return nothing but a list of 3 questions like this ["question 1", "question 2", "question 3"]
        - You must respond the generated questions in Chinese
            """
        prompt = PromptTemplate.from_template(q_prompt)

        qa_chain = prompt|llm|StrOutputParser()|json.loads

        qa_results = qa_chain.invoke({"summary":summary})

        return qa_results
    
    def answer(self, question,summary, content):
        rewrite_query = """ You are a great disruptive thinker in tech industry with extensive experiences in AI, Product Management, and venture capital investment.
        You work with a colleague to answer AI community's hot questions about hot topics.You are given a question\n
        Just like how you read, understand, and think deeply about questions. You approach question critically with a underlying comprehensive thought processes.\n
        Your task is to rewrite the questions to help approach the question from different and comprehensive perspectives and to help your colleague to answer the questions better
        This is the question: \n {question} \
        You are Also given a summary of the content where the question originates from so that you understand the whole context better.
        ------\n
        This is the summary of the content:\n{summary}\n
        ------\n
        ## General Guidance:
        - try to understand the semantic meaning of the questions and construct your thought process with the focus of the what, why, and hows of the question.
        - think about the connections between the summary and your rewrited questions. You rewrited questions should cover more content cohesively and connect the logic behind user's original question
        - Try to use the full information provided
        - Provide 3 rewrited questions in chinese in the format like this ['rewrited question 1','rewrited question 2','rewrited question 3']
        """
        rewrite_prompt = PromptTemplate.from_template(rewrite_query)
        rewrite_chain =  rewrite_prompt|llm|StrOutputParser()
        rewrite_results = rewrite_chain.invoke({"question":question, "summary": summary})

        q_a = """ You are a great disruptive thinker in tech industry with extensive experiences in AI, Product Management, and venture capital investment.\n
        You are given a set of questions and you have to answer the questions using information from the summary and the original content.\n
        Just like how you read, understand, and think deeply about questions. You approach questions critically with a underlying comprehensive thought processes.\n
        The questions help you formulate your thought process and answer the questions critically, comprehensively, and cohesively.\n
        Your Task is to answer the questions comprehensively from different perspectives if needed, using condensed information from summary and full original content if necessary.\n
        ------\n
        These are the questions to answer : \n {questions}\n
        This is the summary of the content:\n{summary}\n
        This is the original content: \n{content}
        ------\n
        ## General Guidance:
        - try to understand the semantic meaning of the questions and construct your thought process with the focus of the what, why, and hows of the question.
        - Answer the question as much as you could and from as many different perspectives as possible
        - Use the full information provided, do not make up answer if you don't know. If you are unsure, just say You don't know
        - Produce a strucutured output in Chinese with Markdown Syntax.
        - do not generate a output with questions being answered one by one. Produce a coherent answer flow with coherence between questions.
            """
        q_a_prompt = PromptTemplate.from_template(q_a)
        q_a_chain = (
            q_a_prompt
            | llm
            | StrOutputParser()
        )

        answer = q_a_chain.invoke({"questions":rewrite_results,"summary":summary,"content":content})
        return answer
        

# Agentic Rag


from typing import Literal,List, Optional,Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# data model  needed
from typing import Literal,List, Optional,Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


# data model needed
class Grade_AI_Relevance(BaseModel):
    """Binary score for whether questions and answers are relevant to AI or product management, including AI industry, the use of AI technology, AI products. and AI product Management."""

    AI_Relevance: Literal['yes','no'] = Field(...,
        description="Questions & Answers are relevant to AI, 'yes' or 'no'"
    )

def grade_ai_relevance(state):
    """
    Determines whether the question is related to AI topics
     Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the grade_question key with binary score 'yes' or 'no'
    """
    print("---CHECK WHETHER QUESTION IS RELATED TO AI")
    question = state['question']
    answer = state['answer']
    structured_llm_grader = llm.with_structured_output(Grade_AI_Relevance)
    system = """You are a grader assessing relevance of a question & answer pair to AI. \n 
    Topics of AI are many, including AI industry, the use of AI technology. AI products, AI investment. AI startups,AI product Management, the career devleopment of AI Product Manager
    Technology includes language model, robotics. AI for Science, computer vision. Autonomous driving AI Agents
    If the question & answer pair contains keyword(s) or semantic meaning related to topics of AI as described above, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the question & answer is relevant to AI."""
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: \n\n {question} \n\n Answer: {answer}"),
    ]
    )
    chain = prompt | structured_llm_grader
    result = chain.invoke({"question":question, "answer":answer})
    result = result.AI_Relevance
    state['ai_relevance'] = result
    return state


class Grade_question(BaseModel):
    """ based on the semantic meaning and key words, categorize the question to one of the following topics
    1. AI Product Management\n
    2. AI State-of-the-art models and techs\n
    3. AI Product Monetization and Go-to-market strategy\n
    4. AI first principle understanding\n
    5. AI Investment\n
    6. AI Industry insights and know-how
    """

    Ai_topics: List[Literal[
            "AI & Product Management and Strategy",
            "AI & State-of-the-art models and techs",
            "AI & first principle understanding",
            "AI & start-ups and VC investments",
            "AI & Industry insights and know-how"
        ]
     ] = Field(
        ...,
        description = "minimum 1 , maximum 2 topics from the predefined set"
    )


def grade_question(state):
    """
    Determines which topics the question is related to in AI
     Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the question_topics key
    """
    print("---CHECK WHICH AI TOPICS QUESTION IS RELATED TO")
    llm_question_grader = llm.with_structured_output(Grade_question)

    # Prompt
    system = """You are a classification assistant with expertise in AI-related topics.
        Your task is to categorize question into specific topics of AI, based on the context provided \n 

        Topics of AI are as follow:\n
        1. AI & Product Management and Strategy\n
        2. AI & State-of-the-art models and techs\n
        3. AI & first principle understanding\n
        4. AI & start-ups and VC investments\n
        5. AI & Industry insights and know-how\n

        choose minimum 1, maximum 2 topics from the predefined topics above to map the question"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "This is the question that needs to be categorized: \n\n {question} \n\n"),
        ]
    )

    chain = prompt | llm_question_grader
    question = state['question']
    result = chain.invoke({"question":question})
    state['question_topics'] = result
    return state



class Grade_response(BaseModel):
    """
    Based on the semantic meaning / key words of the question and answers and the perspectives of the question, 
    decide the binary score for whether the answer contains respoonse  that fit the perspectives the question needs
    If the answer is relevant and cover all the question types , return the binary_score 'yes'.
    If the answer is not relevant nor did not cover the question types, return 'no'
    If the binary_score is 'no', also returns the perspectives that were missed.
    """

    binary_score: Literal['yes', 'no'] = Field(
        ...,
        description="if the answer is adequately good enough"
    )
    missed_topics: Optional[List[Literal[
            "AI & Product Management and Strategy",
            "AI & State-of-the-art models and techs",
            "AI & first principle understanding",
            "AI & start-ups and VC investments",
            "AI & Industry insights and know-how"
    ]]] = Field(
        None,
        description="Categories that were missed, required if binary_score is 'no'")
    suggestions: Optional[str] = Field(None, description = "Suggestions provided for further improvement, required if binary_score is 'no'.")

def grade_reponse(state):
    """
    Determines how good is the answer to the question
     Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the grade_response key
    """
    print("---CHECK WHETHER ORIGINAL ANSWER IS GOOD ENOUGH OR NEEDS REVISION")
    llm_response_grader = llm.with_structured_output(Grade_response)

    # Prompt
    system = """You are a expert answer evaluator that excels at assessing the comprehensivenss of a response to an AI-specific question
        Your task is to determine if the response adequately covers the topic and provides actionable suggestions for improvement if it falls short

        These are some background knowledge about AI for you to better understand the context :\n
        Topics of AI are many, including AI industry, the use of AI technology. AI products, AI investment. AI startups,AI product Management, the career devleopment of AI Product Manager
        Technology includes language model, robotics. AI for Science, computer vision. Autonomous driving AI Agents.

        question: \n
        {question}\n
        topics that the answer needs to cover:\n
        {topics}\n
        This is the original answer:
        {answer}\n

        Criteria for evaluation:\n
        - Relevance: Does the response address the main aspects of the question?\n
        - Depth: Does the response cover all relevant dimensions of the topic, including broader implications and specific details?\n
        - Clarity: Is the response clear and well-structured?\n
        - Accuracy: Is the answer relevant and grounded in facts?\n

        if the above criteria is met, you should return a binary score of 'yes''
        otherwise , you should return no , and provide detailed suggestions on the following aspects:
        - which topics that the answer does not cover ?
        - Additional Information Needed: What specific areas or aspects are missing?
        - Broader Context: Are there broader questions or context that should be included?
        - Detailed Examples or Evidence: What examples or evidence would enhance the response?
        - Clarifications or Revisions: Are there parts of the response that need clearer explanations or revisions?


        potentially missed topicss:\n
        1. AI Product Management\n
        2. AI State-of-the-art models and techs\n
        3. AI Product Monetization and Go-to-market strategy\n
        4. AI first principle understanding\n
        5. AI Investment\n
        6. AI Industry insights and know-how\n
    """
    prompt = PromptTemplate.from_template(system
    )

    chain = prompt | llm_response_grader
    question = state['question']
    answer = state['answer']
    topics=  state['question_topics']
    #docs = retriever.get_relevant_documents(question)
    #doc_txt = docs[1].page_content
    result = chain.invoke({"question": question, "topics":topics, "answer" :answer})
    state['grade_response'] = result.suggestions
    return state

#class step_back_query

def step_back_query(state):
    """
    Generate step-back-questions to help answer revision
     Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the step_back_question key
    """
    print("---GENERATE STEP BACK QUESTIONS")
    system = """ You are an question refinement assistant with expertise in AI. Your task is to take a specific question 
    and extract a more general question that gets at the underlying principles to help answer the specific question in a comprehensive manner
    You will be given a question and a set of ai-topics that this question is related to. 
    Also the suggestions to the original answer is provided. You should use the suggestions to help you generate the questions
    Generate one question for each ai-topics

    General Guidance:
    - general questions generated should capture the underlying themes and concepts
    - Suggestions for further improvements for the answers are provided to help you generate question. You should address the issued mentioned in the suggestions
    - return nothing but a list of generated questions with the corresponding topics. ['topic 1 + question 1', 'topic 2+ question 2']
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ('human', "Original question: {question}\n Topics of AI that question relates to: {topic}\n n suggestions: {suggestion}")
        ]
    )

    chain = prompt|llm|StrOutputParser()
    suggestion = state['grade_response']
    question = state['question']
    topic = state['question_topics']
    result = chain.invoke({"question": question, "topic":topic,"suggestion":suggestion})
    state['step_back_question'] = result
    return state

import pinecone
import os
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def retrieve_from_db(state):
    """
    retrieve from database 
     Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the retrieved_docs key
    """
    print("---RETRIEVE FROM DATABASE")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("ai-pm")
    embed = OpenAIEmbeddings()
    #vector_store = PineconeVectorStore(index = index, embedding = embed)
    #retriever_pinecone = vector_store.as_retriever()
    step_back_question = state['step_back_question']
    result = get_full_docs(step_back_question)
    #result = retriever_pinecone.invoke(step_back_question)
    state['retrieved_docs'] = result

    return state

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def grade_docs_relevance(state):
# LLM with function call
    """
    Grade whether the retrieved docs are useful 
     Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates the docs_relevance
    """
    print("---RETRIEVE FROM DATABASE")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    chain = prompt | structured_llm_grader
    question = state['step_back_question']
    docs = state['retrieved_docs']
    result = chain.invoke({"question": question, "document": docs})
    state['docs_relevance'] = result
    return state


def rag(state):
    """
    perform retrieve generation from database

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates rag_generation key with appended web results
    """

    QA =""" You are an AI expert with comprehensive domain knowledges in different areas of AI, including SOTA model technology, AI Product management, AI start-ups, and AI applications in 
    different industries.\n

    Your task is to provide a detailed and coherent answer to address the questions asked, given the retrieved  AI domain-specific context

    This is the question asked:\n
    {question}\n


    This is the additional Document:\n
    {docs}\n

    Genearl Guidance:
    - Keep your answer ground in the facts of the DOCUMENT.
    - If the DOCUMENT doesn't contain the facts to answer the QUESTION, just return "i don't know"
    - answer shouldbe comprehensive, detailed, and coherent
    """
    rag_prompt = PromptTemplate.from_template(QA)
    docs = state['retrieved_docs']
    step_back_question = state['step_back_question']
    rag_chain = (
                {"docs": docs, "question": RunnablePassthrough()}
                | rag_prompt
                |llm
                |StrOutputParser()
    )

    result= rag_chain.invoke(step_back_question)
    state['rag_generation'] =result

def final_answer_generation(state):
    """
    perform final answer generation using the retrieved docs or search results

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates final_answer key with appended web results
    """
    system = """ You are an AI expert with comprehensive domain knowledges in different areas of AI, including SOTA model technology, AI Product management, AI start-ups, and AI applications in 
    different industries.\n

    Your task is to provide a detailed and coherent answer to address the questions asked, given the retrieved  AI domain-specific context.

    Original Answer\n
    {original_answer}\n
    
    suggestions for improvement:\n
    {suggestion}\n

    Additional useful information, which are retrieved according to the suggestions:\n
    The additional information usually provides useful background information to comprehensively answer the question.\n
    {additional_information}\n

    General Guidance:\n
    - use the provided suggestions to guide the improvements\n
    - incorporate the additional useful information to add depth and detail to the answer.\n
    - Generate the enhanced response:\n
        - Provide a detailed and coherent answer that addresses the question comprehensively.\n
	    - Ensure the response integrates both the original content and additional improvements effectively.\n
    """
    prompt = PromptTemplate.from_template(system)
    chain = prompt|llm|StrOutputParser()
    original_answer = state["answer"]
    suggestion = state['grade_response']
    additional_information = None
    if state['docs_relevance'] == 'yes':
        additional_information = state['rag_generation']
    else:
        additional_information = state['search_results']
    final_answer = chain.invoke({"original_answer":original_answer,"suggestion":suggestion,"additional_information":additional_information})
    state['final_answer'] = final_answer
    return state


from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=5,search_depth = "advanced")

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    #web_results = Document(page_content=web_results)
    state['search_results'] = web_results
    return state


def decide_to_grade_question (state):
    """
    Determines whether to grade the question

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    if state["ai_relevance"] == 'yes':
        return "grade_question"
    
    else:
        return "END"
    
def decide_docs_relevance(state):
    """
    Determines whether to use retrieved docs for generation or use web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    if state["docs_relevance"] == "yes":
        return "rag"
    
    else:
        return "web_search"
    
def decide_good_answer(state):
    """
    Determines whether the original answer is good enough.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    if state["grade_response"] == "yes":
        return "END"
    
    else:
        return "step_back_query"

class RagAgentSTATE(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: original question asked
        answer: original answer 
        ai_relevance: binary score for whether the question is related to AI
        question_topics: list of topics that the question is related to in AI
        grade_response: binary score for whether the answer is good engouh. if the score is no, missed topics and suggestions will also be given
        step_back_question: step back to generate more general questions that provided background information to help formulate the final answer
        documents: list of documents retrieved from database
        search_results: documents fetched via online search
        generation: LLM generation to answer the step_back_question either using documents retrieved or search results
        final_answer: combine generation and original answer to formulate the final_answer
    """

    question: str
    answer: str
    ai_relevance:str
    question_topics:List[str]
    grade_response:Dict[str, any]
    step_back_question:str
    retrieved_docs:str
    docs_relevance:str
    search_results:str
    rag_generation: str
    final_answer:str


from langgraph.graph import END, START
class AgenticRag:
    def __init__(self):
        pass
    def set_up_team(self):
        workflow = StateGraph(RagAgentSTATE)
        workflow.add_node("grade_ai_relevance",grade_ai_relevance)
        workflow.add_node("grade_question",grade_question)
        workflow.add_node("grade_reponse",grade_reponse)
        workflow.add_node("step_back_query",step_back_query)
        workflow.add_node("retrieve_from_db",retrieve_from_db)
        workflow.add_node("grade_docs_relevance",grade_docs_relevance)
        workflow.add_node("web_search",web_search)
        workflow.add_node("rag",rag)
        workflow.add_node("final_answer_generation",final_answer_generation)

        workflow.add_edge(START,"grade_ai_relevance")
        workflow.add_conditional_edges(
            "grade_ai_relevance",
            decide_to_grade_question,
            {
                "grade_question":"grade_question",
             "END":END,},
        )
        workflow.add_edge("grade_question","grade_reponse")
        workflow.add_conditional_edges(
            "grade_reponse",
            decide_good_answer,
            {
                "END":END,
                "step_back_query":"step_back_query",
            },
        )
        workflow.add_edge("step_back_query","retrieve_from_db")
        workflow.add_edge("retrieve_from_db","grade_docs_relevance")
        workflow.add_conditional_edges(
            "grade_docs_relevance",
            decide_docs_relevance,
            {
                "rag":"rag",
             "web_search":"web_search",},
        )
        workflow.add_edge("rag","final_answer_generation")
        workflow.add_edge("web_search","final_answer_generation")

        app = workflow.compile()
        return app
    def run(self):
        chain = self.set_up_team()
        return chain
