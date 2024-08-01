import os
#tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
from dotenv import load_dotenv
os.environ['LANGCHAIN_PROJECT'] = 'Langgraph'
load_dotenv()

import os
import requests
import operator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#gpt-researcher
from gpt_researcher import GPTResearcher

llm = ChatOpenAI(model='gpt-4o-mini')

class ResearchAgent:
    def __init__(self):
        pass
    
    def enter_chain (self,context):
        example_query = [
            "why is Nvdia stock going up?",
            "what are the first principles of AI-first Product Design?",
            "what are strategies to monetize gen AI features in SaaS?",
            "when are we likely to see mass adoption of AI in consumer apps?",
            "where are we in the AI cycles?"
        ]
        prompt = ChatPromptTemplate.from_messages([
            ("human","""You are a research assistant,based on the context given:\n
             ------\n
             {context}\n
             ------\n
             , generate a concise research query with a focus on the topic that worth conducting a 
             in-depth research.\n
             below are some exemplary research queries:\n
             {example_query}
             return nothing but one research query
             """)
        ]
        )
        enter_chain = prompt|llm|StrOutputParser()
        research_query = enter_chain.invoke({"context":context,"example_query":example_query})

        return research_query

    def exit_chain(self,report):
        translate_prompt = ChatPromptTemplate.from_messages([
            ("human","""You are a research report editor, you are given a full English research report\n
             ------\n
             {report}\n
             ------\n
             ## Goal
             your task is to translate the English research report into Chinese.
             return nothing but the translated report.\n
             ## General Guidance:\n
             - Keep the original format and do not change the format
             - keep the content and meaning consistent, you are only doing a translation.
             
             """)
        ]
        )
        exit_chain = translate_prompt|llm|StrOutputParser()
        final_report = exit_chain.invoke({"report":report})
        return final_report

    async def run(self,context) -> str:
        type = "research_report"
        query = self.enter_chain(context)
        researcher = GPTResearcher(query, type)
        research_result = await researcher.conduct_research()
        report = await researcher.write_report()
        final_report = self.exit_chain(report)
        return final_report
    
 
                                




