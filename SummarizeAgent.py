import requests
import os
import ast
import operator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
os.environ['LANGCHAIN_PROJECT'] = 'Langgraph'
load_dotenv()

def calculate_reading_time(word_count, words_per_minute=300):
    # Calculate the reading time in minutes and round to the nearest whole number
    reading_time = round(word_count / words_per_minute)
    return reading_time


#Summarize Agent
def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Look at the conversations closely, if both roles have been selected and they have responded with valid content in messages."
                "Then usually you can go straight to FINISH."
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

llm = ChatOpenAI(model='gpt-4o-mini')
fast_llm = ChatOpenAI(model="gpt-3.5-turbo")
teamsummarize_supervisor_agent = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: Parse_and_scrape, Summarize. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status."
    " Look at the conversastions carefully, if both workers have responded with meaningful content."
    " It means you can respond with FINISH"
    " Don't call the workers repeatedly if they have provided valid response",
    ["Summarize", "Parse_and_scrape"],
)

class Parse_and_scrape:
    def __init__(self):
        pass

    def parser(self,state:dict):

        messages = state["messages"]
        input = messages[0].content
        parse_url_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a powerful assistant ",
                ),
                (
                    "human",
                    "look at the input: {input}, parse out the urls into a list,\n"
                    "return nothing but  a list of urls in the following format:\n"
                    "if there is only one url, return a list like this [\"url\"]\n"
                    "if there are multiple urls, please return like this [\"url1\",\"url2\",\"url3\"]"
                    "Only return the parse list of urls. Do not include any other words in your response",
                    
                )
            ]
        )

        parse_url_chain = parse_url_prompt|fast_llm|StrOutputParser()
        url_lists = parse_url_chain.invoke({"input":input})
        url_lists = ast.literal_eval(url_lists)
        print(f"type of url_lists is {type(url_lists)}")
        return url_lists

    def scraper(self, url_lists:List):
        base_url = "https://r.jina.ai/"
        api_key = os.getenv("JINA_READER_API")
        headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
        scraped_content = []
        counter = 1
        for url in url_lists:    
            api_url = base_url + url
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                content = data['data']

                # Add the description key
                description = (
                     f"This is scraped content No.{counter}."
                     f"原文长度 {len(content.get('content',''))}, 阅读全文需要约{calculate_reading_time(len(content.get('content','')))}分钟"
                )
                content['description'] = description
                scraped_content.append(content)

                counter += 1 
            else:
                print(f"Failed to retrieve data for {url}. Status code: {response.status_code}")
                content = {
                    'title': 'N/A',
                    'url': url,
                    'content': "N/A",
                    'description': f"this should be scraped article No.{counter},but failed to retrieve data for {url}."
                }
                counter += 1 
                scraped_content.append(content)
        return scraped_content
    
    def run(self,state:dict):
        url_lists = self.parser(state)
        scraped_content = self.scraper(url_lists)
        return{"messages":[HumanMessage(str(scraped_content), name = "Parse_and_scrape")]}

class Summarize:
    def __init__(self,option):
        self.option = option
        self.prompt_template = self.get_prompt_template()

    def get_prompt_template(self):
        if self.option == 0:
            return """
            # Goal: \n
            # Please provide a narritive summary for the below article provided\n
            # General Guidelines:\n
            - Focus on the core events, maintaining a sense of the original narrative flow.\n
            - Be concise and Chronological. \n
            - Write with markdown syntax and provide the response in Chinese\n
            - Please do your best, this is very important to my career\n
            ------\n
            article: \n
            {article}
            ------
            """
        elif self.option == 1:
            return """ 
        # Goal: \n    
        Provide a comprehensive summary of each article provided in context.\n
        Pay close attention to the title, url, description as there could be multiple articles provided.\n
        Please produce detailed summary that is at least 50%\ length of the orginal article.\n

        # Requirements: \n
        you should follow the below thought processes to produce best outputs: \n
        - first read through the whole article\n
        - break down the whole article into sections\n
        - identify key points and arguments in each section\n
        - produce informative and comprehensive summaries for key points and arguments in each section\n

        Summary for each article should consist of 2 parts\n
        First part:\n
        You should first produce a descriptive summary about the whole article in one paragraph no longer than 300 characters.\n
        Second part: \n
        Follow the below strucutre to produce summary for each section, and write with maximum depth of information
        1.主题: use one sentence to summarize the section.\n 
        2.观察与发现:  extract the main findings and insights talked about in the section.\
        include opinions from all characters if multiple people express their ideas  \n
        3.原因: focus on the why and how of the insights above, illustrate with maximum details\n
        4.引用: make direct referenes, list evidences inlcuding factual information and numbers, and use quotations from original article that supports the findings and arguments.\n

        # General guidelines: \n
        - Summary should be well strcutured and informative and in depth, with facts and numbers if available and a minimum of 3000 words.\n
        - Keep Writing the summary as long as you can using all relevant and necessary information provided\n
        - Maintain the logical flow, ensuring the summary is coherent and easy to follow\n
        - Write with markdown syntax and provide the response in Chinese\n
        - Sometimes article provided might not contain actual useful information. If so, respond with "no content available"\n
        Please do your best, this is very important to my career\n
        ------\n
        article: \n
        {article}
        ------
        """
        else:
            raise ValueError("Invalid option.")
    
    def produce_summary(self, state:dict):
        messages = state["messages"]
        article = messages[-1].content
        #print(f"""article is {article}""")
        #print(type(article))
        summary_template = self.prompt_template
        prompt = ChatPromptTemplate.from_template(summary_template)
        prompt = ChatPromptTemplate.from_template(summary_template)
        produce_summary_chain = prompt|llm|StrOutputParser()
        summary = produce_summary_chain.invoke({"article":article})
        return summary
    
    def run(self,state:dict):
        summary = self.produce_summary(state)

        return {"messages":[HumanMessage(summary,name="Summarize")]}
    

# The agent state is the input to each node in the graph
class SummarizeTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str

class SummarizerAgent:
    def __init__(self,option):
        self.option = option
        self.Sumamrizer_graph= None

    def init_team(self):
        summarizer = Summarize(self.option)
        Parser = Parse_and_scrape()

        memory = SqliteSaver.from_conn_string(":memory:")
        Summarizer_workflow= StateGraph(SummarizeTeamState)
        Summarizer_workflow.add_node("Parse_and_scrape", Parser.run)
        Summarizer_workflow.add_node("Summarize", summarizer.run)
        Summarizer_workflow.add_node("Summarizer_team", teamsummarize_supervisor_agent)

        members = ["Summarize", "Parse_and_scrape"]

        Summarizer_workflow.add_edge("Parse_and_scrape","Summarize")
        Summarizer_workflow.add_edge("Summarize",END)
        # The supervisor populates the "next" field in the graph state
        # which routes to a node or finishes
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        Summarizer_workflow.add_conditional_edges("Summarizer_team", lambda x: x["next"], conditional_map)
        # Finally, add entrypoint
        Summarizer_workflow.set_entry_point("Summarizer_team")

        self.Summarizer_graph = Summarizer_workflow.compile(checkpointer= memory)

        return self.Summarizer_graph
    
    
    def run(self, messages, thread_id):
        thread_id =1
        self.config = {"configurable": {"thread_id":thread_id}}
        self.Summarizer_graph = self.init_team()
        results = self.Summarizer_graph.stream({"messages":[HumanMessage(content=messages)]},self.config)
        return results

    def get_state_messages(self):
        all_states = []
        for state in self.Summarizer_graph.get_state_history(self.config):
            all_states.append(state)
        counter = 0
        info_list = []
        for messages in all_states[0].values["messages"]:
            counter += 1
            if counter == 2:
                content =ast.literal_eval(messages.content)
                for i in content:
                    info_dict={}
                    title = i.get("title")
                    url = i.get("url")
                    description = i.get("description")
                    info_dict["title"] = title
                    info_dict["url"] = url
                    info_dict["description"] = description
                    info_list.append(info_dict)

        return info_list