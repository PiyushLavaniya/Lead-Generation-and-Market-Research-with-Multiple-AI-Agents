from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import csv

load_dotenv()

##Instantiating an LLM
llm = ChatGroq(model = "llama-3.1-70b-versatile", temperature = 0)

##Instantiating the TOOL
search_tool = SerperDevTool()

#### Can add agents to search for the linked profile info of the people given by the lead researcher to get more information.
#### Can add an agent to search linkedin based on the compnaies given by the lead researcher agent.
#### Can pass context from one task to another task to provide more contextual info while an agent is performing a task.
#### Can define the number of leads to find while researching for leads.
#### Can save the Agents' outputs at each step/task.
#### Have some information about the services our company provides.

##Creating Agents

##First, Lead Researcher Agent
lead_researcher = Agent(
    role = "Lead Researcher",
    goal = "Find Potential leads based on industry, job title, and location.",
    backstory = "You are an expert in finding business leads for marketing and sales outreach.",
    llm = llm,
    verbose = True,
    allow_delegation = False,
)

##Second, Lead Scorer Agent
lead_scorer = Agent(
    role = "Lead Scorer",
    goal = "Score leads based on their potential value and relevance to our business.",
    backstory = "You are adapt at evaluating business leads, identifying their potential value for our company.",
    llm = llm,
    verbose = True,
    allow_delegation = False
)

##Third, Market Researcher Agent
market_researcher = Agent(
    role = "Market Researcher",
    goal = "Add relevant market insights for each lead to help the sales team understand the current trends.",
    backstory = "You provide cirtical market insights for better lead targetting.",
    llm = llm,
    verbose = True,
    allow_delegation = False,
)

##Fourth, Competitor Analyst Agent (Cna provide the tool to look around the web)
competitor_analyst = Agent(
    role = "Competitor Analyst",
    goal = "Analyze competitors of the lead's business and add insights to enhance the outreach.",
    backstory = "You are an expert at analyzing market competition to refine sales approaches.",
    llm = llm,
    tools = [search_tool],
    verbose = True,
    allow_delegation = False
)

##Fifth, Data Validator Agent
data_validator = Agent(
    role = "Data Validator",
    goal = "Ensure the leads are accurate and meet the specified criteria.",
    backstory = "You cross-check leads to guarantee quality and relevance for the sales team.",
    llm = llm,
    verbose = True,
    allow_delegation = False
)

##Sixth, Lead Segmentation Agent
lead_segmentation = Agent(
    role = "Lead Segmentation Specialist",
    goal = "Group leads into relevant segments for more targeted marketing and outreach.",
    backstory = "You organize and categorize leads to optimize campaign strategies.",
    llm = llm,
    verbose = True,
    allow_delegation = False
)

##Seventh, Email Writer Agent
email_writer = Agent(
    role = "Email Writer",
    goal = "Write personalized emails for each validated lead and save them to a CSV file.",
    backstory = "You are a skilled communicator, crafting compelling emails that resonate with leads.",
    llm = llm,
    verbose = True,
    allow_delegation = False
)

##Eighth, Report Generator Agent
report_generator = Agent(
    role = "Report Generator",
    goal = "Compile the validated leads into a structured report for the sales team.",
    backstory = "Your ensure that the sales team gets a clean, organized report of all leads.",
    llm = llm,
    verbose = True,
    allow_delegation = False
)

##Creating Tasks
##First, Lead Research Task for Lead Researcher Agent (Can add Job Title)
lead_research_task = Task(
    description = ("Research and Identify 3 potential business leads based on the criteria such as Industry, CEO and location."
                   "Indsutry: {industry}." 
                   "Location: {location}."
                   "Use available web resources and databases to gather this information."),
    expected_output = "A list of leads with relevant contact details and basic company information.",
    agent = lead_researcher,
    tools = [search_tool],
)

##Second, Lead Scoring Task for Lead Scorer Agent (Need to give Business Info)
lead_scoring_task = Task(
    description = "Evaluate each lead and assign a score based on its relevance to the business and potential value.",
    expected_output = "A score between 1 and 100 for each lead.",
    agent = lead_scorer
)

##Third, Market Research Task for Market Researcher Agent
market_research_task = Task(
    description = "Conduct market research for each lead to identify current trends and opportunities.",
    expected_output = "Market Insights relevant to the lead's industry.",
    agent = market_researcher,
    tools = [search_tool],
)

##Fourth, Competitor Analysis Task for Competitor Analyst (Can provide our business' info to find out potential competitors for the leads using Web Search or LinkedIn)
competitor_analysis_task = Task(
    description = "Analyze the lead's competitors and provide insights that can inform outreach strategies.",
    expected_output = "A competitor analysis for each lead.",
    agent = competitor_analyst
)

##Fifth, Lead Validation Task for Lead Validator (Can provide specified criteria and the web search capability)
lead_validation_task = Task(
    description = "Validate the accuracy of the leads and ensure that they meet the specified criteria.",
    expected_output = "A verified list of leads.",
    agent = data_validator,
    context = [lead_scoring_task, lead_research_task]
)

##Sixth, Lead Segmentation Task for Lead Segmentation Agent
lead_segmentation_task = Task(
    description = "Segment the leads based on industry, company size, or other relevant categories.",
    expected_output = "A segmented list of leads grouped by category.",
    agent = lead_segmentation,
    context = [lead_scoring_task, lead_validation_task]
)

##Seventh, Email Writing Task for Eamil Writer Agent
email_writing_task = Task(
    description = "Write a personalized email to each lead and save the lead details and email content in a CSV file.",
    expected_output = "A CSV file with the lead details and the personalized email content.",
    agent = email_writer,
    context = [lead_research_task, lead_segmentation_task],
    output_file = "Leads-and-Emails.csv"
)

##Eighth, Report Generation Task for Report Generator Agent
Report_generation_task = Task(
    description = "Compile all the validated leads and their details into a structured report for the sales team.",
    expected_output = "A detailed report of all leads, segmented and validated.",
    agent = report_generator,
    output_file = "Report-on-leads.md",
    context = [lead_segmentation_task, lead_research_task]
)

##Creating the Crew with the Agents and respective Tasks
lead_generation_crew = Crew(
    agents = [lead_researcher, lead_scorer, market_researcher, competitor_analyst, data_validator, lead_segmentation, email_writer, report_generator],
    tasks = [lead_research_task, lead_scoring_task, market_research_task, competitor_analysis_task, lead_validation_task, lead_segmentation_task, email_writing_task, Report_generation_task],
    process = Process.sequential
)

##Kicking off this Crew
crew_result = lead_generation_crew.kickoff(inputs = {
    "industry": "Textile",
    "job_title": "CTO",
    "location": "India"
})