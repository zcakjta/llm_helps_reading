from AipmAgent import AipmAgent,further_qa,AgenticRag
from ResearchAgent import ResearchAgent
from SummarizeAgent import SummarizerAgent
import streamlit as st
import streamlit_antd_components as sac
from streamlit_carousel import carousel
from streamlit_gsheets import GSheetsConnection
from streamlit_timeline import timeline
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import pandas as pd
import os 
import asyncio

from dotenv import load_dotenv
# configure for langsmith tracking
os.environ['LANGCHAIN_PROJECT'] = 'Langgraph'
load_dotenv()

#declare functions that make aysnc run work in streamlit
def run_async_func(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args))
    loop.close()
    return result

# page config
st.set_page_config(page_title="📖llm-helps-reading_v0.01", page_icon='📖',layout='wide')
st.title('🤓GPT阅读总结助手_v0.0.1')
st.caption('💬 让阅读更加高效\nPowered By Streamlit & OpenAI')

# side bars

css = """
<style>
.gradient-text-research {
    background: linear-gradient(to right, orange, brown);
    -webkit-background-clip: text;
    color: transparent;
    font-weight: bold;
}

.gradient-text-product {
    background: linear-gradient(to right, orange, brown);
    -webkit-background-clip: text;
    color: transparent;
    font-weight: bold;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
image = Image.open('out-0.png')
with st.sidebar:
    st.sidebar.image(image, use_column_width=True)
    tabs = sac.tabs([
        sac.TabsItem(label='产品功能介绍'),
        sac.TabsItem(label='直接体验 Agents', tag ="暂未开放",disabled= True)
    ], align='center',use_container_width=True, return_index=True)
    if tabs == 0:
        st.markdown("**应用主要使用以下能力**:")
        st.markdown("""**总结功能**: _快速总结_ (约300字, 用于文章预读);  _深度总结_ (提取文章核心观点和逻辑); 一次可执行多个url的内容抓取和总结.""")
        st.markdown("""- 使用[Jina Reader](https://jina.ai/reader/)爬取网页内容并转化成适合大模型处理的数据格式\n- 使用[Langgraph](https://langchain-ai.github.io/langgraph/)构建人机交互的路由能力.
                    """)
    
        st.markdown(" ")

        st.markdown('<span class="gradient-text-research">**研究专家 Agent**</span>: 基于文章主题内容进行深入研究，提供更丰富的行业insights', unsafe_allow_html=True)
        st.markdown("""- 使用[Tavily AI](https://tavily.com/)实现在线搜索.\n- 使用[GPT-Researcher](https://gptr.dev/)实现高质量研究报告的编写.
                    """)

        st.markdown(" ")

        st.markdown('<span class="gradient-text-product">**产品专家 Agent**</span>: 基于高质量 AI 知识库，针对文章主题进行 RAG 问答，提取文章看点等能力', unsafe_allow_html=True)
        st.markdown("""- 知识库主要内容来源于 知识星球- AI产品经理大本营(by Hanniman).\n- 使用[STORM](https://arxiv.org/html/2401.18059v1)方法通过树搜索以提升召回的覆盖率\n- 使用[Pinecone](https://www.pinecone.io/) 向量数据库, [Cohere Rerank](https://cohere.com/rerank)等API搭建高效的 RAG 链路.
                    """)   
    
    elif tabs == 1:
        description_text ="""体验完整功能，请从右侧开始 👉\n\n----------\n\n直接与Agents交互, 请从下方开始 👇"""
        sac.alert(label='Agent 使用说明', banner=False,description=description_text, icon=True,size='md',closable=True,color='blue')
    
        researcher_prompt = st.text_area(
                    ":blue[**请输入 AI 研究内容选题**:]", height = 150,
                    placeholder="武汉萝卜快跑爆火的现象对Robotaxi意味着什么?\n\n按 Enter 换行，按 Ctr+Enter 提交") 

        aipm_prompt = st.text_area(":orange[**请向 AI 产品专家提问**:]",height = 150,
                                            placeholder="AI原生的产品设计方法论是什么?\n\n按 Enter 换行，按 Ctr+Enter 提交"
                                            )
    
        if researcher_prompt:
            Researcher_agent_trial = ResearchAgent()
            Reseacher_response_trial =run_async_func(Researcher_agent_trial.run, researcher_prompt)

        if aipm_prompt:
            Aipm_agent_trial = AipmAgent()
            Aipm_response_trial = Aipm_agent_trial.run()
             
# initilize the session states
session_keys = ["submitted",
                "empty_containers"
                "user_input",
                "title",
                "url",
                "article", 
                "description",
                "summary", 
                "full_response",
                "research_report", 
                "extracted_opinions",
                "aipm",
                "researcher",
                "further_qa",
                "qa_lists",
                "question_asked",
                "answer_to_question_asked"]
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = []

# main page when
add_vertical_space(5)
with st.form('my_form'):
            input_text = st.text_area(label= '',label_visibility= "collapsed",height = 150,
                                    placeholder = '''想要总结的网址都丢进来\n\n想要总结的段落都复制进来\n\n按 Enter 换行，按 Ctr+Enter 提交''')
            
            submitted = st.form_submit_button('提交',type="primary")
option = sac.segmented(
                    items=[
                        sac.SegmentedItem(label='⚡️  快速总结'),
                        sac.SegmentedItem(label='🕰️  深度总结')
                    ], align='center', color='red',return_index = True)

divider_container = st.empty()
divider_c1, divider_c2, divider_c3 = divider_container.columns(3)
with divider_c2:
        sac.divider(label='💡来点阅读灵感', align='center', size='xs', variant='dotted', color='gray')
dataframe_container = st.empty()
below_c1, below_c2 = dataframe_container.columns(2)
with below_c1:
        add_vertical_space(3)   
        df = pd.DataFrame({
        "推荐阅读": ["Andrew Karpathy 主页", "Microsoft Graph RAG", "Llama 3.1"],
        
        "网址": ["https://karpathy.ai/.", "https://microsoft.github.io/graphrag/", "https://ai.meta.com/blog/meta-llama-3-1/"]
        
    })

        st.dataframe(df,
                    column_config={
            "Title": "Title ",
            "Notes": st.column_config.LinkColumn("Notes"),
            "Categories": st.column_config.SelectboxColumn(
                "categories",
                width="medium",
                options=[
                    "📊 AI Agents",
                    "📈 AI Product",
                    "🤖 AI Models",
                ],
                required=True,
            )
        },
        hide_index=True,
    )

    # connect to G-sheet for automatic updated infos
with below_c2:
        url ="https://docs.google.com/spreadsheets/d/1pRmQxJRmMBjMIaz546FX5hvHxG8ra4AJYAZUOj_uR_U/edit?usp=sharing"
        con = st.connection("gsheets", type=GSheetsConnection)
        data = con.read(spreadsheet=url)
        def create_carousel_item(df):
            carousel_item = []
            for i, row in df.iterrows():
                carousel_item.append(
                    dict(
                        title="",
                        text="",
                        img=row['img'],
                        link=row['url']
                    )
                )
            return carousel_item
        carousel(items= create_carousel_item(data),interval = 3000,controls =False,container_height = 200)

if submitted:
        st.session_state["submitted"] = True
        st.session_state["empty_containers"] = True
        divider_container.empty()
        dataframe_container.empty()
        with st.spinner('✍🏼阅读助手总结中....'):
            Summarizer_agent = SummarizerAgent(option)
            response  = Summarizer_agent.run(input_text,2)
            article = None
            summary = None
            for event in response:
                if 'Parse_and_scrape' in event:
                    article = event["Parse_and_scrape"]["messages"][0].content
                if 'Summarize' in event:
                    summary = event["Summarize"]["messages"][0].content
            info_list = Summarizer_agent.get_state_messages()
            st.session_state["user_input"] = input_text
            st.session_state["artcile"] = article
            st.session_state["summary"] = summary
            st.session_state['title'] = [item["title"] for item in info_list if "title" in item]
            st.session_state['url'] = [item["url"] for item in info_list if "url" in item]
            st.session_state['description'] = [item["description"] for item in info_list if "description" in item]
    
        if st.session_state["artcile"] and st.session_state["summary"] :

                descriptions = st.session_state['description']
                titles = st.session_state['title']
                urls = st.session_state['url']
                summary =st.session_state["summary"] 
                markdown_content = ""
    # Iterate over the lists with their indices
                for index, (title, url, description) in enumerate(zip(titles, urls, descriptions)):
                    if index == len(titles) - 1:  # Check if this is the last item
                        # If it is the last item, append the summary as well
                        markdown_content += f"📖 [{title}]({url}); 文章信息： {description}\n\n{summary}\n"
                    else:
                        # If it is not the last item, just append the current title, url, and description
                        markdown_content += f"📖 [{title}]({url}); 文章信息： {description}\n\n"
                # display the content within the container        
                #st.markdown(markdown_content)

                st.session_state["full_response"] = markdown_content

def pop_up():
    popup_widget = sac.segmented(
    items=[
        sac.SegmentedItem(label='看点提取'),
        sac.SegmentedItem(label='深入研究'),
        sac.SegmentedItem(label='',disabled=True),
        sac.SegmentedItem(label='',disabled=True),
        sac.SegmentedItem(label='追问文章内容'),
    ], label='', align='center', use_container_width=True,color='red', 
    bg_color='transparent', divider=False,return_index=True
)
    if popup_widget == 0:
        st.session_state["aipm"] = True
        st.session_state["researcher"] = False
        st.session_state['further_qa'] = False

    elif popup_widget == 1:
        st.session_state["researcher"] = True
        st.session_state["aipm"] = False
        st.session_state['further_qa'] = False

    elif popup_widget  == 4:
        st.session_state["further_qa"] = True
        st.session_state["researcher"] = False
        st.session_state['aipm'] = False

if st.session_state["submitted"]:
    #if st.session_state["empty_containers"]:
    divider_container.empty()
    dataframe_container.empty()
    st.subheader("阅读笔记", divider="gray")
    with st.container(border=True,height = 800):
         st.markdown(st.session_state["full_response"])
         # display further functions  
    pop_up()
    agentic_rag = AgenticRag()
    agentic_rag_response = agentic_rag.run()
       
    if st.session_state["aipm"]:
        Aipm_agent =AipmAgent()
        Aipm_response = Aipm_agent.run()
        summary = st.session_state["summary"]
        article = st.session_state["artcile"]  

        if st.button("产品专家 - 提取看点 🔍"):
            st.divider()
            for s in Aipm_response.stream({"summary":summary,"original_context":article}):
                if "qa" in s:
                    extracted_opinions = s["qa"]["novel_points"]
                    st.session_state["extract_opinions"] = extracted_opinions
                    st.markdown(extracted_opinions)

    if  st.session_state["researcher"]:
        
        Researcher_agent = ResearchAgent()
        summary = st.session_state["summary"]

        if st.button("研究专家 - 生成主题研究报告 📰"):
            st.spinner("生成主题研究报告中....")
            st.divider()
            Researcher_response = run_async_func(Researcher_agent.run,summary)
            st.session_state["research_report"] = Researcher_response
            st.markdown(Researcher_response)

    if  st.session_state["further_qa"]:
        #if not st.session_state["qa_lists"]:
        qa_agent = further_qa()
        summary = st.session_state["summary"]
        follow_up_questions = qa_agent.question(summary)
        st.session_state["qa_lists"] = follow_up_questions
        q_buttons =[st.button(question, use_container_width=True) for question in follow_up_questions]
        #else:
            #follow_up_questions = st.session_state["qa_lists"]
            #q_buttons =[st.button(question, use_container_width=True) for question in follow_up_questions]
        
        for i, clicked in enumerate(q_buttons):
            if clicked:
                question = st.session_state["qa_lists"][i]
                st.session_state["question_asked"] = question
                content = st.session_state["article"]
                summary = st.session_state["summary"]
                answer = further_qa().answer(question,summary,content)
                st.session_state["answer_to_question_asked"]= answer
                st.subheader(question,divider="gray")
                st.markdown(answer)
                question = st.session_state["question_asked"]
                answer = st.session_state["answer_to_question_asked"]
                for s in agentic_rag_response.stream({"question":question,"answer":answer}):
                    if "web_search" in s:
                            st.spinner("启动AI搜索中....")
                                    
                    elif "final_answer_generation" in s:
                        enhanced_response = s['final_answer_generation']['final_answer']
                        expander =st.expander("结合AI搜索，生成答案")
                        expander.write(enhanced_response)

        
    if prompt:= st.chat_input("💭 想深入了解文章内容？请提问"):
            qa_agent = further_qa()
            question = prompt
            content = st.session_state["article"]
            summary = st.session_state["summary"]
            answer_to_question = further_qa().answer(question,summary,content)
            st.subheader(f"{prompt}", divider="gray")
            st.write(answer_to_question)


